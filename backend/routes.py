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
)
from guidance import (
    generate_instruction, generate_guidance_command, stabilize_instruction,
)

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
    config.TARGET_OBJECT = str(raw_target).strip() or "cell phone"
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
    if target_info is not None:
        draw_target_highlight(frame, target_info)
    elif config.yolo_cache is not None:
        # 锁定目标失败时再显示全部红框，避免平时界面过于拥挤
        draw_yolo_detections(frame, config.yolo_cache)

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

    # ---- 生成原始指令 ----
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
                config.TARGET_OBJECT = str(msg["target"]).strip() or config.TARGET_OBJECT
                logger.info(f"目标已通过 WS 更新为: {config.TARGET_OBJECT}")
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
