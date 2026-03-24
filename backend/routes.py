"""HTTP and WebSocket route handlers."""
# HTTP 和 WebSocket 路由处理器

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import config
from models import yolo_model, hands, depth_model
from vision import (
    decode_frame, resize_frame, encode_frame, rotate_frame,
    draw_yolo_detections, draw_hand_landmarks, draw_target_highlight,
    find_best_target, compute_hand_center, estimate_depth, depth_guidance_reliable,
    detect_target_object, detect_hand, draw_annotations,
    get_all_detections, FrameGrabber,
    update_grab_detection, reset_grab_state, draw_grab_success,
)
from guidance import (
    generate_instruction, generate_guidance_command, stabilize_instruction,
)
from voice import parse_intent, IntentResult

logger = logging.getLogger(__name__)

router = APIRouter()

# 单线程推理执行器，保证模型调用线程安全
_inference_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# HTTP 接口
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    return {"message": "视障人士视觉辅助系统 - 后端已运行"}


@router.post("/set-target")
async def set_target(target: dict):
    """设置目标物体"""
    raw_target = target.get("target", "cell phone")
    new_target = str(raw_target).strip() or "cell phone"
    if new_target != config.TARGET_OBJECT:
        config.TARGET_OBJECT = new_target
        reset_grab_state()
    return {"status": "success", "target": config.TARGET_OBJECT}


@router.post("/set-rotation")
async def set_rotation(body: dict):
    """设置画面旋转角度 (0, 90, 180, 270)"""
    deg = int(body.get("rotation", 0))
    if deg not in (0, 90, 180, 270):
        deg = 0
    config.ROTATION = deg
    return {"status": "success", "rotation": config.ROTATION}


# ---------------------------------------------------------------------------
# 帧处理核心函数（在线程池中执行，不阻塞事件循环）
# ---------------------------------------------------------------------------

def _process_frame_sync(frame):
    """
    同步处理单帧画面，所有 CPU 密集型推理集中在这里：
    旋转 → 缩放 → YOLO检测 → 手部检测 → 深度估计 → 指令生成 → 编码
    返回可直接发送给前端的 JSON 字典。
    """
    # ---- 旋转画面 ----
    if config.ROTATION != 0:
        frame = rotate_frame(frame, config.ROTATION)

    # ---- 缩放到处理宽度（降低分辨率 = 大幅提速） ----
    frame = resize_frame(frame, target_width=config.PROCESS_WIDTH)
    # 深度估计使用无标注原图，避免绘制框线污染深度结果
    inference_frame = frame.copy()

    # ---- YOLOv8 物体检测（按间隔执行，其余帧复用缓存） ----
    config.frame_count += 1
    if yolo_model is not None:
        if config.frame_count % config.YOLO_INTERVAL == 1 or config.yolo_cache is None:
            try:
                config.yolo_cache = yolo_model(inference_frame, verbose=False, device='cpu')
            except Exception as e:
                logger.warning(f"YOLOv8 推理出错: {e}")

    # ---- MediaPipe 手部检测 ----
    hand_results = None
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        hand_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        draw_hand_landmarks(frame, hand_results)
    except Exception as e:
        logger.warning(f"MediaPipe 手部检测出错: {e}")

    # ---- 目标锁定（带时序跟踪 + 遮挡兜底） ----
    h, w = frame.shape[:2]
    target_info = find_best_target(
        config.yolo_cache,
        config.TARGET_OBJECT,
        hand_results=hand_results,
        frame_shape=(h, w),
    )

    hand_center = compute_hand_center(hand_results, w, h)

    # ---- 深度估计（高频执行非常耗 CPU，按 DEPTH_INTERVAL 间隔执行） ----
    if (config.DEPTH_INTERVAL > 0
            and depth_model is not None
            and target_info is not None
            and hand_center is not None
            and config.frame_count % config.DEPTH_INTERVAL == 1):
        config.depth_cache = estimate_depth(inference_frame)

    # 深度软门控：可靠时按默认阈值，不可靠时抬高阈值但不完全禁用
    current_depth_map = None
    current_depth_threshold = config.DEPTH_THRESHOLD
    if target_info and hand_center and config.depth_cache is not None:
        current_depth_map = config.depth_cache
        if not depth_guidance_reliable(config.depth_cache, hand_center, target_info):
            current_depth_threshold = max(config.DEPTH_THRESHOLD * 1.8, 0.05)

    # ---- 抓取检测（基于锁定位置 + 手部覆盖 + 深度校验） ----
    grab_state = update_grab_detection(
        target_info=target_info,
        hand_results=hand_results,
        frame_shape=(h, w),
        depth_map=config.depth_cache,
    )

    # ---- 根据抓取状态绘制 ----
    if grab_state == "grabbed":
        draw_grab_success(frame, config.locked_target_bbox)
    elif target_info is not None:
        draw_target_highlight(frame, target_info)
    elif config.locked_target_bbox is not None and grab_state == "close":
        # 目标丢失但手正在覆盖，显示锁定位置
        draw_target_highlight(frame, {
            "x1": config.locked_target_bbox[0],
            "y1": config.locked_target_bbox[1],
            "x2": config.locked_target_bbox[2],
            "y2": config.locked_target_bbox[3],
            "class_name": config.TARGET_OBJECT,
            "confidence": 0.0,
        })
    elif config.yolo_cache is not None:
        # 锁定目标失败时再显示全部红框，避免平时界面过于拥挤
        draw_yolo_detections(frame, config.yolo_cache)

    # ---- 生成原始指令（抓取成功时直接覆盖） ----
    if grab_state == "grabbed":
        raw_instruction = "Object Grabbed!"
    elif grab_state == "close":
        raw_instruction = "Grab it now!"
    else:
        raw_instruction = generate_instruction(
            target_info, hand_center,
            config.TARGET_OBJECT, config.COMMAND_THRESHOLD,
            depth_map=current_depth_map,
            depth_threshold=current_depth_threshold,
        )

    # ---- 防抖 + 冷却 ----
    instruction_to_send = stabilize_instruction(raw_instruction)

    # ---- 检测到的物体列表（去重） ----
    all_objects = get_all_detections(config.yolo_cache)
    config.detected_objects = all_objects

    # ---- 编码帧为 Base64 JPEG ----
    encoded = encode_frame(frame)

    return {
        "image": encoded,
        "instruction": raw_instruction,
        "display_instruction": raw_instruction,
        "speech_instruction": instruction_to_send,
        "detected_objects": [obj["name"] for obj in all_objects],
        "target": config.TARGET_OBJECT,
        "grab_state": grab_state,
    }


# ---------------------------------------------------------------------------
# WebSocket 消息接收（后台协程，独立于主推送循环）
# ---------------------------------------------------------------------------

async def _ws_receive_loop(websocket: WebSocket):
    """
    后台协程：持续监听前端 WebSocket 消息（目标切换、旋转等）。
    独立于视频推送主循环运行，避免轮询开销。
    """
    try:
        while True:
            msg = await websocket.receive_json()
            if "target" in msg:
                new_target = str(msg["target"]).strip() or config.TARGET_OBJECT
                if new_target != config.TARGET_OBJECT:
                    config.TARGET_OBJECT = new_target
                    reset_grab_state()
                    logger.info(f"目标已通过 WS 更新为: {config.TARGET_OBJECT}，抓取状态已重置")
            if "rotation" in msg:
                deg = int(msg["rotation"])
                if deg in (0, 90, 180, 270):
                    config.ROTATION = deg
    except (WebSocketDisconnect, Exception):
        pass


# ---------------------------------------------------------------------------
# WebSocket — DroidCam 拉流 (/ws/video)
# ---------------------------------------------------------------------------

@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket 视频流端点，三大优化消除延迟：
    1. FrameGrabber 后台线程持续拓取最新帧（丢弃旧帧，消除缓冲堆积）
    2. 推理管线在单线程执行器中运行（不阻塞事件循环）
    3. 帧率限制避免前端积压
    """
    await websocket.accept()
    logger.info("WebSocket /ws/video 客户端已连接")

    # 新连接时重置抓取状态
    reset_grab_state()

    # 启动帧拓取器（后台线程持续读取，只保留最新帧）
    grabber = FrameGrabber(config.DROIDCAM_URL)
    if not grabber.is_opened:
        logger.error(f"无法打开 DroidCam 视频流: {config.DROIDCAM_URL}")
        await websocket.send_json({
            "image": None,
            "instruction": "Cannot open DroidCam stream",
        })
        await websocket.close()
        return

    grabber.start()
    logger.info(f"DroidCam 帧拓取器已启动: {config.DROIDCAM_URL}")

    loop = asyncio.get_event_loop()
    # 启动后台消息接收协程（目标/旋转切换）
    recv_task = asyncio.create_task(_ws_receive_loop(websocket))

    try:
        while True:
            # 获取最新帧（不会拿到旧帧，因为 grabber 持续丢弃旧帧）
            frame = grabber.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # 在单线程推理执行器中处理帧（线程安全，不阻塞事件循环）
            result = await loop.run_in_executor(
                _inference_executor, _process_frame_sync, frame
            )

            # 发送处理结果到前端
            await websocket.send_json(result)

            # 帧率限制：避免处理太快导致前端积压
            await asyncio.sleep(1.0 / config.TARGET_FPS)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/video 客户端已断开")
    except Exception as e:
        logger.error(f"WebSocket /ws/video 错误: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        recv_task.cancel()
        grabber.stop()
        logger.info("DroidCam 帧拓取器已停止")


# ---------------------------------------------------------------------------
# WebSocket — 客户端上传帧 (/ws/guidance)
# ---------------------------------------------------------------------------

@router.websocket("/ws/guidance")
async def websocket_endpoint(websocket: WebSocket):
    """客户端上传帧的 WebSocket 实时通信端点"""
    await websocket.accept()
    logger.info("WebSocket 客户端已连接")

    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data.get("frame")

            if not frame_data:
                continue

            frame = decode_frame(frame_data)
            if frame is None:
                continue

            # 检测目标物体和手部
            target_info = detect_target_object(frame, config.TARGET_OBJECT)
            hand_info = detect_hand(frame)

            # 生成指导指令
            guidance = generate_guidance_command(target_info, hand_info)

            # 绘制标注
            annotated_frame = draw_annotations(frame, target_info, hand_info)

            # 编码回传帧
            frame_encoded = encode_frame(annotated_frame)

            response = {
                "image": frame_encoded,
                "command": guidance["command"],
                "message": guidance["message"],
                "target": config.TARGET_OBJECT,
                "frame_info": {
                    "target": target_info if target_info else None,
                    "hand": {k: v for k, v in hand_info.items() if k != "landmarks"},
                },
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端已断开连接")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.close()


# ---------------------------------------------------------------------------
# WebSocket — 语音交互 (/ws/voice)
# ---------------------------------------------------------------------------

# 全局唤醒状态
_voice_awake = False


def _generate_voice_response(intent: IntentResult, current_instruction: str) -> dict:
    """
    根据意图生成语音响应
    回答完一个问题后自动休眠，等待下次唤醒
    
    Returns:
        dict: {
            'response': str,      # 要播放的响应文本
            'action': str,        # 执行的动作
            'target': str | None, # 选择的目标
            'awake': bool         # 是否处于唤醒状态
        }
    """
    global _voice_awake
    
    result = {
        'response': '',
        'action': 'none',
        'target': None,
        'awake': False  # 默认回答后休眠
    }
    
    if intent.intent_type == IntentResult.WAKE:
        _voice_awake = True
        result['awake'] = True  # 唤醒时保持唤醒
        result['response'] = "I'm listening"
        result['action'] = 'wake'
        logger.info("语音唤醒成功")
    
    elif intent.intent_type == IntentResult.SELECT_TARGET:
        if intent.target:
            config.TARGET_OBJECT = intent.target
            result['target'] = intent.target
            result['response'] = f"Tracking {intent.target}"
            result['action'] = 'select_target'
            logger.info(f"语音选择目标: {intent.target}")
        else:
            result['response'] = "I didn't catch that."
        # 回答后休眠
        _voice_awake = False
    
    elif intent.intent_type == IntentResult.QUERY_LOCATION:
        target_name = intent.target or config.TARGET_OBJECT
        if target_name:
            if current_instruction and current_instruction not in ["", "Hand not found"]:
                if "not found" in current_instruction.lower():
                    result['response'] = f"I can't see the {target_name}."
                else:
                    result['response'] = current_instruction
            else:
                result['response'] = f"Looking for {target_name}."
        else:
            result['response'] = "What object?"
        result['action'] = 'query_location'
        # 回答后休眠
        _voice_awake = False
    
    elif intent.intent_type == IntentResult.QUERY_GRASP:
        if current_instruction:
            if "grasp" in current_instruction.lower():
                result['response'] = "Yes, you got it!"
            elif "not found" in current_instruction.lower():
                result['response'] = "I don't see it."
            else:
                result['response'] = "Not yet."
        else:
            result['response'] = "Not tracking."
        result['action'] = 'query_grasp'
        # 回答后休眠
        _voice_awake = False
    
    elif intent.intent_type == IntentResult.STOP:
        _voice_awake = False
        result['response'] = "Okay."
        result['action'] = 'stop'
        logger.info("语音停止/休眠")
    
    else:
        if _voice_awake:
            result['response'] = "Say an object name or ask where it is."
            # 没理解时也休眠
            _voice_awake = False
    
    return result


@router.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    语音交互 WebSocket 端点（简化版）
    
    语音识别由浏览器 Web Speech API 完成，后端只处理意图解析
    
    客户端发送:
    - {"type": "text", "text": "recognized text"}  # 浏览器识别的文本
    - {"type": "wake"}   # 唤醒信号
    - {"type": "sleep"}  # 休眠信号
    
    服务端响应:
    - {"type": "intent", "intent": str, "response": str, "target": str, "awake": bool}
    """
    global _voice_awake
    
    await websocket.accept()
    logger.info("WebSocket /ws/voice 客户端已连接")
    
    try:
        # 发送初始状态
        await websocket.send_json({
            'type': 'status',
            'awake': _voice_awake,
            'message': 'Voice system ready. Say "Hey Tom" to wake me up.'
        })
        
        while True:
            data = await websocket.receive_json()
            msg_type = data.get('type', '')
            
            if msg_type == 'wake':
                _voice_awake = True
                logger.info("语音唤醒")
                continue
            
            if msg_type == 'sleep':
                _voice_awake = False
                logger.info("语音休眠")
                continue
            
            if msg_type == 'text':
                text = data.get('text', '')
                if not text:
                    continue
                
                logger.info(f"收到语音文本: {text}")
                
                # 解析意图
                intent = parse_intent(
                    text,
                    detected_objects=config.detected_objects,
                    is_awake=_voice_awake
                )
                
                # 获取当前指令
                current_instruction = getattr(config, 'last_raw_instruction', '')
                
                # 生成响应
                response = _generate_voice_response(intent, current_instruction)
                
                await websocket.send_json({
                    'type': 'intent',
                    'intent': intent.intent_type,
                    'raw_text': intent.raw_text,
                    'response': response['response'],
                    'action': response['action'],
                    'target': response['target'],
                    'awake': response['awake']
                })
            
            elif msg_type == 'ping':
                await websocket.send_json({'type': 'pong'})
    
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/voice 客户端已断开")
    except Exception as e:
        logger.error(f"WebSocket /ws/voice 错误: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
