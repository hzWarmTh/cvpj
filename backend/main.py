import asyncio
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)
from ultralytics import YOLO
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 应用初始化
app = FastAPI(title="视障人士视觉辅助系统 - 后端")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
TARGET_OBJECT = "cell phone"  # 默认目标物体
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISTANCE_THRESHOLD = 50  # 像素阈值

# DroidCam IP 摄像头地址
DROIDCAM_URL = 'http://192.168.31.126:4747/video'

# ----- 帧计数 & YOLO 缓存（每 N 帧才跑一次 YOLO） -----
frame_count = 0
YOLO_INTERVAL = 5          # 每 5 帧执行一次 YOLO
yolo_cache = None           # 缓存上一次 YOLO 检测结果

# ----- YOLO 绘制参数 (红色) -----
YOLO_BOX_COLOR = (0, 0, 255)
YOLO_TEXT_COLOR = (0, 0, 255)
YOLO_BOX_THICKNESS = 2
YOLO_FONT_SCALE = 0.6

# ----- MediaPipe 手部骨骼绘制参数 (绿色) -----
HAND_LANDMARK_COLOR = (0, 255, 0)
HAND_CONNECTION_COLOR = (0, 255, 0)
HAND_LANDMARK_RADIUS = 4
HAND_CONNECTION_THICKNESS = 2

# ----- 目标物体高亮绘制参数 (蓝色) -----
TARGET_BOX_COLOR = (255, 0, 0)        # 蓝色 (BGR)
TARGET_BOX_THICKNESS = 3
TARGET_FONT_SCALE = 0.7

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 强制使用 CPU，避免显卡兼容性问题
DEVICE = 'cpu'
logger.info(f"强制使用计算设备: {DEVICE}")

# 初始化 YOLOv8 模型（加载到 CPU）
try:
    model = YOLO('yolov8n.pt')  # 使用纳米版本以提高速度
    model.to('cpu')
    logger.info("YOLOv8 模型加载成功，运行设备: cpu")
except Exception as e:
    logger.error(f"YOLOv8 模型加载失败: {e}")
    model = None

# ---------------------------------------------------------------------------
# MiDaS Small 单目深度估计模型（用于前后方向判断）
# ---------------------------------------------------------------------------
depth_model = None
depth_transform = None

try:
    depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    depth_model.eval()
    depth_model.to('cpu')

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    depth_transform = midas_transforms.small_transform

    logger.info("MiDaS Small 深度估计模型加载成功 (CPU)")
except Exception as e:
    logger.warning(f"MiDaS 模型加载失败（将跳过前后判断）: {e}")
    depth_model = None
    depth_transform = None

# ----- 深度阈值 -----
DEPTH_THRESHOLD = 0.03  # 归一化深度差阈值（需根据实际场景微调）

# ---------------------------------------------------------------------------
# 指令冷却 & 防抖状态变量
# ---------------------------------------------------------------------------
import time

last_raw_instruction = ""          # 上一帧计算出的原始指令
stable_instruction_count = 0       # 当前指令连续出现的帧数
STABLE_THRESHOLD = 10              # 连续多少帧才认为稳定

last_sent_instruction = ""         # 上一次实际发送给前端的指令
last_sent_time = 0.0               # 上一次发送的时间戳
COOLDOWN_SECONDS = 3.0             # 相同指令重复发送的冷却时间（秒）


def decode_frame(frame_data: str) -> np.ndarray:
    """解码 Base64 帧数据"""
    try:
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"解码帧失败: {e}")
        return None


def resize_frame(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
    """将帧缩放到指定宽度，保持宽高比"""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)


def encode_frame(frame: np.ndarray) -> str:
    """编码帧为 Base64 字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    except Exception as e:
        logger.error(f"编码帧失败: {e}")
        return None


def draw_yolo_detections(frame: np.ndarray, results) -> None:
    """在画面上绘制 YOLOv8 检测到的物体边界框和标签（红色）"""
    if results is None or len(results) == 0:
        return
    detections = results[0]
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_BOX_COLOR, YOLO_BOX_THICKNESS)

        label = f"{class_name} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, YOLO_FONT_SCALE, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            YOLO_BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, YOLO_FONT_SCALE,
            (255, 255, 255), 1, cv2.LINE_AA,
        )


def draw_hand_landmarks(frame: np.ndarray, hand_results) -> None:
    """在画面上绘制 MediaPipe 手部关键点和骨骼连线（绿色）"""
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return
    h, w, _ = frame.shape
    for hand_landmarks in hand_results.multi_hand_landmarks:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_point, end_point,
                     HAND_CONNECTION_COLOR, HAND_CONNECTION_THICKNESS, cv2.LINE_AA)
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), HAND_LANDMARK_RADIUS,
                       HAND_LANDMARK_COLOR, cv2.FILLED, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 空间计算 & 指令生成（用于 /ws/video）
# ---------------------------------------------------------------------------
COMMAND_THRESHOLD = 50  # 像素阈值


def estimate_depth(frame: np.ndarray) -> np.ndarray | None:
    """
    使用 MiDaS Small 对当前帧进行单目深度估计。
    返回归一化深度图 (H_depth, W_depth)，值越大表示离摄像头越近。
    注意：输出分辨率与原图不同，需要坐标映射。
    """
    if depth_model is None or depth_transform is None:
        return None
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = depth_transform(img_rgb).to('cpu')

        with torch.no_grad():
            prediction = depth_model(input_batch)

        # 插值回原图尺寸，确保坐标一一对应
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 归一化到 0~1（实测：值越小 = 离摄像头越近，值越大 = 离摄像头越远）
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map
    except Exception as e:
        logger.warning(f"深度估计失败: {e}")
        return None


def get_depth_at(depth_map: np.ndarray, cx: float, cy: float,
                 patch_size: int = 10) -> float:
    """
    取 (cx, cy) 周围 patch_size×patch_size 区域的平均深度值，
    避免单像素噪声影响。depth_map 已插值到原图尺寸，无需额外映射。
    """
    h, w = depth_map.shape
    x, y = int(cx), int(cy)
    half = patch_size // 2
    x1 = max(0, x - half)
    x2 = min(w, x + half)
    y1 = max(0, y - half)
    y2 = min(h, y + half)
    return float(depth_map[y1:y2, x1:x2].mean())


def find_best_target(yolo_results, target_name: str = "cup") -> dict | None:
    """
    从 YOLO 检测结果中筛选置信度最高的目标物体。
    返回 {x1, y1, x2, y2, cx, cy, confidence, class_name} 或 None。
    """
    if yolo_results is None or len(yolo_results) == 0:
        return None
    detections = yolo_results[0]
    best = None
    best_conf = 0.0
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        if target_name.lower() not in class_name.lower():
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "confidence": conf, "class_name": class_name,
            }
    return best


def find_best_target_any(yolo_results) -> dict | None:
    """
    [调试用] 不过滤类别，返回置信度最高的任意物体。
    """
    if yolo_results is None or len(yolo_results) == 0:
        return None
    detections = yolo_results[0]
    best = None
    best_conf = 0.0
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "confidence": conf, "class_name": class_name,
            }
    return best


def log_all_detections(yolo_results) -> list[str]:
    """
    [调试用] 将当前帧所有检测到的物体名称打印到终端，并返回名称列表。
    """
    if yolo_results is None or len(yolo_results) == 0:
        return []
    detections = yolo_results[0]
    names = []
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        conf = float(box.conf[0])
        names.append(f"{class_name}({conf:.2f})")
    if names:
        logger.info(f"[DEBUG] 当前帧检测到的物体: {', '.join(names)}")
    return names


def compute_hand_center(hand_results, frame_w: int, frame_h: int) -> dict | None:
    """
    取第一只手所有关键点的平均像素坐标作为手部中心。
    返回 {cx, cy} 或 None。
    """
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return None
    lm = hand_results.multi_hand_landmarks[0].landmark
    cx = sum(p.x for p in lm) / len(lm) * frame_w
    cy = sum(p.y for p in lm) / len(lm) * frame_h
    return {"cx": cx, "cy": cy}


def generate_command(target: dict | None, hand: dict | None,
                     threshold: int = COMMAND_THRESHOLD) -> dict:
    """
    根据手部与目标的相对位置生成指令。
    返回 {command, dx, dy, distance}。
    """
    if target is None:
        return {"command": "Target not found", "dx": None, "dy": None, "distance": None}
    if hand is None:
        return {"command": "Hand not found", "dx": None, "dy": None, "distance": None}

    dx = hand["cx"] - target["cx"]
    dy = hand["cy"] - target["cy"]
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if abs(dx) <= threshold and abs(dy) <= threshold:
        cmd = "Grasp"
    elif abs(dx) > abs(dy):
        cmd = "Move Left" if dx > 0 else "Move Right"
    else:
        cmd = "Move Up" if dy > 0 else "Move Down"

    return {
        "command": cmd,
        "dx": round(dx, 1),
        "dy": round(dy, 1),
        "distance": round(distance, 1),
    }


def generate_instruction(target: dict | None, hand: dict | None,
                         target_name: str = TARGET_OBJECT,
                         threshold: int = COMMAND_THRESHOLD,
                         depth_map: np.ndarray = None,
                         depth_threshold: float = DEPTH_THRESHOLD) -> str:
    """
    根据目标与手部的空间关系生成指令字符串。
    优先级：Z轴（前后） > X/Y轴（左右上下）
    先保证深度合适（手不遮挡物体），再指导平面对齐。
    """
    if target is None:
        return f"Target {target_name} not found"
    if hand is None:
        return "Hand not found"

    # ---------- 第一层：Z轴/深度判断（优先） ----------
    if depth_map is not None:
        depth_hand = get_depth_at(depth_map, hand["cx"], hand["cy"])
        depth_obj = get_depth_at(depth_map, target["cx"], target["cy"])
        # MiDaS 实测: 值越小 = 离摄像头越近，值越大 = 离摄像头越远
        # diff > 0 → 手的深度值更大 → 手离摄像头更远（手在物体后面） → "Move Forward"
        # diff < 0 → 手的深度值更小 → 手离摄像头更近（手挡住物体） → "Move Backward"
        diff = depth_hand - depth_obj
        print(f"DEBUG: Hand Depth={depth_hand:.2f}, Obj Depth={depth_obj:.2f}, Diff={diff:.2f}")
        logger.debug(f"[DEPTH] hand={depth_hand:.3f} obj={depth_obj:.3f} diff={diff:.3f}")

        if diff > depth_threshold:
            print(f"Logic: Hand is farther (depth_hand={depth_hand:.2f} > depth_obj={depth_obj:.2f}), returning 'Move Forward'")
            return "Move Forward"
        elif diff < -depth_threshold:
            print(f"Logic: Hand is closer (depth_hand={depth_hand:.2f} < depth_obj={depth_obj:.2f}), returning 'Move Backward'")
            return "Move Backward"

    # ---------- 第二层：X/Y轴对齐（深度合适后） ----------
    # 第一人称 + 手机背面摄像头：
    # dx = target_x - hand_x > 0 代表目标在手的右侧，应提示 Move Right
    dx = target["cx"] - hand["cx"]   # obj_x - hand_x
    dy = target["cy"] - hand["cy"]   # obj_y - hand_y

    if abs(dx) <= threshold and abs(dy) <= threshold:
        return "Grasp"

    # 优先处理绝对值较大的偏差
    if abs(dx) >= abs(dy):
        if dx > threshold:
            return "Move Right"
        elif dx < -threshold:
            return "Move Left"
    # Y 轴：图像坐标系 Y 向下为正
    if dy > threshold:
        return "Move Down"
    elif dy < -threshold:
        return "Move Up"

    return "Grasp"


def draw_target_highlight(frame: np.ndarray, target: dict) -> None:
    """
    用蓝色框高亮绘制目标物体（区别于红色的普通检测框）。
    """
    x1, y1 = int(target["x1"]), int(target["y1"])
    x2, y2 = int(target["x2"]), int(target["y2"])

    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  TARGET_BOX_COLOR, TARGET_BOX_THICKNESS)

    label = f"[TARGET] {target['class_name']} {target['confidence']:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, TARGET_FONT_SCALE, 2
    )
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6),
                  (x1 + tw, y1), TARGET_BOX_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (x1, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, TARGET_FONT_SCALE,
                (255, 255, 255), 2, cv2.LINE_AA)


def detect_target_object(frame: np.ndarray, target_name: str) -> dict:
    """检测目标物体"""
    if model is None:
        return None
    
    try:
        results = model(frame, conf=0.5, verbose=False, device='cpu')
        
        for result in results:
            for detection in result.boxes:
                class_name = result.names[int(detection.cls)]
                
                if target_name.lower() in class_name.lower():
                    x1, y1, x2, y2 = detection.xyxy[0]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    confidence = detection.conf[0]
                    
                    return {
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidence),
                        "class": class_name
                    }
        
        return None
    except Exception as e:
        logger.error(f"目标检测失败: {e}")
        return None


def detect_hand(frame: np.ndarray) -> dict:
    """检测手部关键点"""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 计算手部中心 (使用手腕关键点)
            wrist = hand_landmarks.landmark[0]
            center_x = wrist.x * frame.shape[1]
            center_y = wrist.y * frame.shape[0]
            
            return {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "landmarks": hand_landmarks,
                "detected": True
            }
        
        return {"detected": False}
    except Exception as e:
        logger.error(f"手部检测失败: {e}")
        return {"detected": False}


def generate_guidance_command(target_info: dict, hand_info: dict) -> dict:
    """生成语音指导指令"""
    if not target_info or not hand_info.get("detected"):
        return {
            "command": "wait",
            "message": "正在等待检测目标物体和手部..."
        }
    
    target_x = target_info["center_x"]
    target_y = target_info["center_y"]
    hand_x = hand_info["center_x"]
    hand_y = hand_info["center_y"]
    
    dx = hand_x - target_x
    dy = hand_y - target_y
    distance = (dx**2 + dy**2) ** 0.5
    
    # 判断相对位置并生成指令
    if distance < DISTANCE_THRESHOLD:
        return {
            "command": "grasp",
            "message": "距离足够，现在可以抓取了！",
            "distance": float(distance)
        }
    
    # 判断移动方向
    if abs(dx) > abs(dy):
        if dx > 0:
            return {
                "command": "move_left",
                "message": "请向左移动你的手。",
                "distance": float(distance)
            }
        else:
            return {
                "command": "move_right",
                "message": "请向右移动你的手。",
                "distance": float(distance)
            }
    else:
        if dy > 0:
            return {
                "command": "move_up",
                "message": "请向上移动你的手。",
                "distance": float(distance)
            }
        else:
            return {
                "command": "move_down",
                "message": "请向下移动你的手。",
                "distance": float(distance)
            }


def draw_annotations(frame: np.ndarray, target_info: dict, hand_info: dict) -> np.ndarray:
    """在帧上绘制标注"""
    annotated_frame = frame.copy()
    
    # 绘制目标物体边界框
    if target_info:
        cv2.rectangle(
            annotated_frame,
            (int(target_info["x1"]), int(target_info["y1"])),
            (int(target_info["x2"]), int(target_info["y2"])),
            (0, 255, 0), 2
        )
        cv2.circle(
            annotated_frame,
            (int(target_info["center_x"]), int(target_info["center_y"])),
            5, (0, 255, 0), -1
        )
        cv2.putText(
            annotated_frame,
            f"Target: {target_info['class']}",
            (int(target_info["x1"]), int(target_info["y1"]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    # 绘制手部
    if hand_info.get("detected") and hand_info.get("landmarks"):
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            frame_rgb,
            hand_info["landmarks"],
            mp_hands.HAND_CONNECTIONS
        )
        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    return annotated_frame


@app.get("/")
async def root():
    return {"message": "视障人士视觉辅助系统 - 后端已运行"}


@app.post("/set-target")
async def set_target(target: dict):
    """设置目标物体"""
    global TARGET_OBJECT
    raw_target = target.get("target", "cell phone")
    TARGET_OBJECT = str(raw_target).strip() or "cell phone"
    return {"status": "success", "target": TARGET_OBJECT}


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket 视频流端点：从 DroidCam 拉取画面，检测后推送标注图片 + 指令"""
    await websocket.accept()
    logger.info("WebSocket /ws/video 客户端已连接")

    loop = asyncio.get_event_loop()
    cap = cv2.VideoCapture(DROIDCAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error(f"无法打开 DroidCam 视频流: {DROIDCAM_URL}")
        await websocket.send_json({"image": None,
                                   "instruction": "Cannot open DroidCam stream"})
        await websocket.close()
        return

    logger.info(f"DroidCam 视频流已打开: {DROIDCAM_URL}")

    try:
        while True:
            # 在线程池中读取帧，避免阻塞事件循环
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret or frame is None:
                await asyncio.sleep(0.05)
                continue

            # 缩放到 640 宽度以加速检测
            frame = resize_frame(frame, target_width=640)

            # ---------- YOLOv8 物体检测 ----------
            global frame_count, yolo_cache
            frame_count += 1

            if model is not None:
                if frame_count % YOLO_INTERVAL == 1 or yolo_cache is None:
                    try:
                        yolo_results = model(frame, verbose=False, device='cpu')
                        yolo_cache = yolo_results
                    except Exception as e:
                        logger.warning(f"YOLOv8 推理出错: {e}")

                # 绘制所有检测（红色框）
                if yolo_cache is not None:
                    draw_yolo_detections(frame, yolo_cache)

            # ---------- 目标锁定：找到置信度最高的 TARGET_OBJECT ----------
            h, w = frame.shape[:2]
            target_info = find_best_target(yolo_cache, TARGET_OBJECT)

            # 用蓝色框高亮目标物体
            if target_info is not None:
                draw_target_highlight(frame, target_info)

            # ---------- MediaPipe 手部检测 ----------
            hand_results = None
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                hand_results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True
                draw_hand_landmarks(frame, hand_results)
            except Exception as e:
                logger.warning(f"MediaPipe 手部检测出错: {e}")

            hand_center = compute_hand_center(hand_results, w, h)

            # ---------- 深度估计（当目标和手都检测到时运行） ----------
            current_depth_map = None
            if depth_model is not None and target_info is not None and hand_center is not None:
                current_depth_map = estimate_depth(frame)

            # ---------- 生成原始指令（含前后判断） ----------
            raw_instruction = generate_instruction(
                target_info, hand_center,
                TARGET_OBJECT, COMMAND_THRESHOLD,
                depth_map=current_depth_map,
                depth_threshold=DEPTH_THRESHOLD,
            )

            # ---------- 第二步：稳定性过滤（防抖） ----------
            global last_raw_instruction, stable_instruction_count
            global last_sent_instruction, last_sent_time

            if raw_instruction == last_raw_instruction:
                stable_instruction_count += 1
            else:
                stable_instruction_count = 1
                last_raw_instruction = raw_instruction

            # ---------- 第三步：冷却时间判断 ----------
            instruction_to_send = ""  # 仅用于语音播报

            if stable_instruction_count >= STABLE_THRESHOLD:
                # 指令已稳定
                current_time = time.time()

                if raw_instruction != last_sent_instruction:
                    # 指令变了，立即发送
                    instruction_to_send = raw_instruction
                    last_sent_instruction = raw_instruction
                    last_sent_time = current_time
                else:
                    # 指令未变，检查冷却时间
                    if current_time - last_sent_time > COOLDOWN_SECONDS:
                        # 超过冷却时间，再次提醒
                        instruction_to_send = raw_instruction
                        last_sent_time = current_time
                    # 否则保持沉默（instruction_to_send = "")

            # ---------- 返回 JSON ----------
            encoded = encode_frame(frame)
            await websocket.send_json({
                "image": encoded,
                # 始终显示当前指令
                "instruction": raw_instruction,
                "display_instruction": raw_instruction,
                # 仅在稳定并通过冷却后才播报
                "speech_instruction": instruction_to_send,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/video 客户端已断开")
    except Exception as e:
        logger.error(f"WebSocket /ws/video 错误: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        cap.release()
        logger.info("DroidCam 视频流已释放")


@app.websocket("/ws/guidance")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 实时通信端点"""
    await websocket.accept()
    logger.info("WebSocket 客户端已连接")
    
    try:
        while True:
            # 接收客户端数据
            data = await websocket.receive_json()
            frame_data = data.get("frame")
            
            if not frame_data:
                continue
            
            # 解码帧
            frame = decode_frame(frame_data)
            if frame is None:
                continue
            
            # 检测目标物体和手部
            target_info = detect_target_object(frame, TARGET_OBJECT)
            hand_info = detect_hand(frame)
            
            # 生成指导指令
            guidance = generate_guidance_command(target_info, hand_info)
            
            # 绘制标注
            annotated_frame = draw_annotations(frame, target_info, hand_info)
            
            # 编码回传帧
            frame_encoded = encode_frame(annotated_frame)
            
            # 准备响应数据
            response = {
                "image": frame_encoded,
                "command": guidance["command"],
                "message": guidance["message"],
                "target": TARGET_OBJECT,
                "frame_info": {
                    "target": target_info if target_info else None,
                    "hand": {k: v for k, v in hand_info.items() if k != "landmarks"}
                }
            }
            
            # 发送响应
            await websocket.send_json(response)
    
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端已断开连接")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await websocket.close()


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    hands.close()
    logger.info("应用已关闭，资源已释放")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
