"""Computer vision utilities: frame encoding, drawing, detection helpers, depth estimation."""
# 计算机视觉工具：帧编解码、绘制、检测辅助、深度估计

import base64
import logging
import threading

import cv2
import numpy as np
import torch

import config
from models import (
    yolo_model, hands, mp_hands, mp_drawing,
    depth_model, depth_transform,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameGrabber：后台线程持续拓取最新帧，消除 DroidCam 缓冲延迟
# ---------------------------------------------------------------------------

class FrameGrabber:
    """
    后台线程持续从摄像头（DroidCam）拓取帧，只保留最新一帧。
    主处理循环每次取到的都是实时画面，彻底消除 OpenCV 视频流缓冲堆积导致的 2-3 秒延迟。
    """

    def __init__(self, url: str):
        self._cap = cv2.VideoCapture(url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def start(self):
        """启动后台拓取线程"""
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        """持续读取帧，丢弃旧帧，只保留最新的"""
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        """获取最新一帧（如果还没有新帧则返回 None）"""
        with self._lock:
            frame = self._frame
            self._frame = None  # 标记已消费
            return frame

    def stop(self):
        """停止拓取并释放资源"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._cap.release()


# ---------------------------------------------------------------------------
# Frame encode / decode / resize
# ---------------------------------------------------------------------------

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
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{frame_base64}"
    except Exception as e:
        logger.error(f"编码帧失败: {e}")
        return None


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """按 0/90/180/270 度旋转帧"""
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      config.YOLO_BOX_COLOR, config.YOLO_BOX_THICKNESS)

        label = f"{class_name} {confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.YOLO_FONT_SCALE, 1,
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            config.YOLO_BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, config.YOLO_FONT_SCALE,
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
                     config.HAND_CONNECTION_COLOR, config.HAND_CONNECTION_THICKNESS,
                     cv2.LINE_AA)
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), config.HAND_LANDMARK_RADIUS,
                       config.HAND_LANDMARK_COLOR, cv2.FILLED, cv2.LINE_AA)


def draw_target_highlight(frame: np.ndarray, target: dict) -> None:
    """用蓝色框高亮绘制目标物体（区别于红色的普通检测框）"""
    x1, y1 = int(target["x1"]), int(target["y1"])
    x2, y2 = int(target["x2"]), int(target["y2"])

    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  config.TARGET_BOX_COLOR, config.TARGET_BOX_THICKNESS)

    label = f"[TARGET] {target['class_name']} {target['confidence']:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.TARGET_FONT_SCALE, 2,
    )
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6),
                  (x1 + tw, y1), config.TARGET_BOX_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (x1, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, config.TARGET_FONT_SCALE,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_annotations(frame: np.ndarray, target_info: dict, hand_info: dict) -> np.ndarray:
    """在帧上绘制标注（用于 /ws/guidance 端点）"""
    annotated_frame = frame.copy()

    # 绘制目标物体边界框
    if target_info:
        cv2.rectangle(
            annotated_frame,
            (int(target_info["x1"]), int(target_info["y1"])),
            (int(target_info["x2"]), int(target_info["y2"])),
            (0, 255, 0), 2,
        )
        cv2.circle(
            annotated_frame,
            (int(target_info["center_x"]), int(target_info["center_y"])),
            5, (0, 255, 0), -1,
        )
        cv2.putText(
            annotated_frame,
            f"Target: {target_info['class']}",
            (int(target_info["x1"]), int(target_info["y1"]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    # 绘制手部
    if hand_info.get("detected") and hand_info.get("landmarks"):
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            frame_rgb,
            hand_info["landmarks"],
            mp_hands.HAND_CONNECTIONS,
        )
        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    return annotated_frame


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

def estimate_depth(frame: np.ndarray) -> np.ndarray | None:
    """
    使用 MiDaS Small 对当前帧进行单目深度估计。
    返回归一化深度图 (H, W)，值越小 = 离摄像头越近，值越大 = 越远。
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

        # 归一化到 0~1
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
    """取 (cx, cy) 周围 patch 区域的平均深度值，避免单像素噪声影响。"""
    h, w = depth_map.shape
    x, y = int(cx), int(cy)
    half = patch_size // 2
    x1 = max(0, x - half)
    x2 = min(w, x + half)
    y1 = max(0, y - half)
    y2 = min(h, y + half)
    return float(depth_map[y1:y2, x1:x2].mean())


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def find_best_target(yolo_results, target_name: str = "cup") -> dict | None:
    """从 YOLO 检测结果中筛选置信度最高的目标物体。"""
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
    """[调试用] 不过滤类别，返回置信度最高的任意物体。"""
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
    """[调试用] 将当前帧所有检测到的物体名称打印到终端。"""
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


def get_all_detections(yolo_results) -> list[dict]:
    """返回当前帧检测到的所有物体的去重列表（按置信度降序）。"""
    if yolo_results is None or len(yolo_results) == 0:
        return []
    detections = yolo_results[0]
    seen = {}
    for box in detections.boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        conf = float(box.conf[0])
        if class_name not in seen or conf > seen[class_name]:
            seen[class_name] = conf
    result = [{"name": name, "confidence": round(conf, 2)}
              for name, conf in seen.items()]
    result.sort(key=lambda x: x["confidence"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Hand center
# ---------------------------------------------------------------------------

def compute_hand_center(hand_results, frame_w: int, frame_h: int) -> dict | None:
    """取第一只手所有关键点的平均像素坐标作为手部中心。"""
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return None
    lm = hand_results.multi_hand_landmarks[0].landmark
    cx = sum(p.x for p in lm) / len(lm) * frame_w
    cy = sum(p.y for p in lm) / len(lm) * frame_h
    return {"cx": cx, "cy": cy}


# ---------------------------------------------------------------------------
# Legacy standalone detection (used by /ws/guidance)
# ---------------------------------------------------------------------------

def detect_target_object(frame: np.ndarray, target_name: str) -> dict:
    """检测目标物体（用于 /ws/guidance 端点）"""
    if yolo_model is None:
        return None

    try:
        results = yolo_model(frame, conf=0.5, verbose=False, device='cpu')

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
                        "class": class_name,
                    }

        return None
    except Exception as e:
        logger.error(f"目标检测失败: {e}")
        return None


def detect_hand(frame: np.ndarray) -> dict:
    """检测手部关键点（用于 /ws/guidance 端点）"""
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
                "detected": True,
            }

        return {"detected": False}
    except Exception as e:
        logger.error(f"手部检测失败: {e}")
        return {"detected": False}
