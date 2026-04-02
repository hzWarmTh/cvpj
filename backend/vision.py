"""Computer vision utilities: frame encoding, drawing, detection helpers, depth estimation."""
# 计算机视觉工具：帧编解码、绘制、检测辅助、深度估计

import base64
import logging
import threading
from typing import Optional

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
# Runtime state for robust target/depth estimation
# ---------------------------------------------------------------------------

_target_track_state = {
    "target_name": None,
    "class_name": None,
    "bbox": None,
    "velocity": (0.0, 0.0),
    "confidence": 0.0,
    "missed": 0,
}

_depth_ema_cache = None

_TRACK_MAX_MISSED = 8
_TRACK_EMA = 0.65
_TRACK_VEL_EMA = 0.7
_TRACK_VEL_DAMP = 0.9
_OCCLUSION_IOU_THRESH = 0.12
_OCCLUSION_CENTER_RATIO = 0.55
_DEPTH_EMA_ALPHA = 0.35
_DEPTH_PERCENTILE_LOW = 2.0
_DEPTH_PERCENTILE_HIGH = 98.0


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
    返回归一化深度图 (H, W)，值越大 = 离摄像头越近，值越小 = 越远。
    """
    if depth_model is None or depth_transform is None:
        return None
    try:
        global _depth_ema_cache

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

        # 使用分位数做鲁棒归一化，避免极端值导致尺度抖动
        d_low = float(np.percentile(depth_map, _DEPTH_PERCENTILE_LOW))
        d_high = float(np.percentile(depth_map, _DEPTH_PERCENTILE_HIGH))
        if d_high - d_low > 1e-6:
            depth_map = np.clip(depth_map, d_low, d_high)
            depth_map = (depth_map - d_low) / (d_high - d_low)
        else:
            depth_map = np.zeros_like(depth_map)

        # 轻量空间平滑，减少局部噪声
        depth_map = cv2.GaussianBlur(depth_map.astype(np.float32), (5, 5), 0)

        # 时间维平滑，抑制帧间闪烁
        if _depth_ema_cache is None or _depth_ema_cache.shape != depth_map.shape:
            _depth_ema_cache = depth_map
        else:
            _depth_ema_cache = (
                _DEPTH_EMA_ALPHA * depth_map
                + (1.0 - _DEPTH_EMA_ALPHA) * _depth_ema_cache
            )

        depth_map = _depth_ema_cache

        return depth_map
    except Exception as e:
        logger.warning(f"深度估计失败: {e}")
        return None


def get_depth_at(depth_map: np.ndarray, cx: float, cy: float,
                 patch_size: int = 10) -> float:
    """取 (cx, cy) 周围 patch 区域的稳健深度值（截尾 + 中位数）。"""
    h, w = depth_map.shape
    x, y = int(cx), int(cy)
    half = patch_size // 2
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)

    patch = depth_map[y1:y2, x1:x2]
    values = patch[np.isfinite(patch)].reshape(-1)
    if values.size == 0:
        return float("nan")

    if values.size < 6:
        return float(np.median(values))

    q_low, q_high = np.percentile(values, [10, 90])
    trimmed = values[(values >= q_low) & (values <= q_high)]
    if trimmed.size == 0:
        trimmed = values

    return float(np.median(trimmed))


def depth_guidance_reliable(depth_map: np.ndarray, hand: dict, target: dict,
                           patch_size: int = 14,
                           std_threshold: float = 0.12,
                           min_valid_ratio: float = 0.65) -> bool:
    """评估手与目标附近深度是否稳定可靠，避免前后指令误判。"""
    if depth_map is None or hand is None or target is None:
        return False

    h, w = depth_map.shape

    def _patch_stats(cx: float, cy: float):
        x = int(cx)
        y = int(cy)
        half = patch_size // 2
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        patch = depth_map[y1:y2, x1:x2]
        vals = patch[np.isfinite(patch)].reshape(-1)
        total = patch.size if patch.size > 0 else 1
        valid_ratio = float(vals.size / total)
        if vals.size == 0:
            return valid_ratio, float("inf")
        return valid_ratio, float(np.std(vals))

    hand_ratio, hand_std = _patch_stats(hand["cx"], hand["cy"])
    tgt_ratio, tgt_std = _patch_stats(target["cx"], target["cy"])

    return (
        hand_ratio >= min_valid_ratio
        and tgt_ratio >= min_valid_ratio
        and hand_std <= std_threshold
        and tgt_std <= std_threshold
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_diag(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _clamp_bbox(bbox: tuple[float, float, float, float],
                frame_shape: Optional[tuple[int, int]]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if frame_shape is None:
        return bbox
    h, w = frame_shape
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _hand_bbox_from_results(hand_results,
                            frame_shape: Optional[tuple[int, int]]) -> tuple[float, float, float, float] | None:
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return None
    if frame_shape is None:
        return None

    h, w = frame_shape
    hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
    xs = [lm.x * w for lm in hand_landmarks]
    ys = [lm.y * h for lm in hand_landmarks]
    if not xs or not ys:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # 给手框加一点外扩，覆盖手掌运动模糊区域
    pad_x = 0.12 * max(1.0, x2 - x1)
    pad_y = 0.12 * max(1.0, y2 - y1)
    bbox = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    return _clamp_bbox(bbox, frame_shape)


def _target_from_bbox(bbox: tuple[float, float, float, float],
                      class_name: str,
                      confidence: float,
                      tracked: bool,
                      occluded: bool) -> dict:
    x1, y1, x2, y2 = bbox
    cx, cy = _bbox_center(bbox)
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": cx, "cy": cy,
        "confidence": confidence,
        "class_name": class_name,
        "tracked": tracked,
        "occluded": occluded,
    }


def find_best_target(yolo_results, target_name: str = "cup",
                     hand_results=None,
                     frame_shape: Optional[tuple[int, int]] = None) -> dict | None:
    """筛选目标并做短时跟踪；遮挡时使用预测位置避免瞬时丢失。"""
    global _target_track_state

    normalized_target = (target_name or "").strip().lower()
    if _target_track_state["target_name"] != normalized_target:
        _target_track_state = {
            "target_name": normalized_target,
            "class_name": None,
            "bbox": None,
            "velocity": (0.0, 0.0),
            "confidence": 0.0,
            "missed": 0,
        }

    if yolo_results is None or len(yolo_results) == 0:
        yolo_boxes = []
        detections = None
    else:
        detections = yolo_results[0]
        yolo_boxes = list(detections.boxes)

    best = None
    best_score = -1.0
    prev_bbox = _target_track_state["bbox"]

    for box in yolo_boxes:
        class_id = int(box.cls[0])
        class_name = detections.names[class_id]
        if normalized_target and normalized_target not in class_name.lower():
            continue

        conf = float(box.conf[0])
        bbox = tuple(float(v) for v in box.xyxy[0].tolist())

        # 结合轨迹连续性评分，减少目标切换
        continuity_bonus = 0.0
        if prev_bbox is not None:
            prev_cx, prev_cy = _bbox_center(prev_bbox)
            cur_cx, cur_cy = _bbox_center(bbox)
            dist = ((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2) ** 0.5
            norm = max(1.0, _bbox_diag(prev_bbox))
            continuity_bonus = max(0.0, 1.0 - dist / (2.5 * norm)) * 0.25

        score = conf + continuity_bonus
        if score > best_score:
            best_score = score
            best = {
                "bbox": _clamp_bbox(bbox, frame_shape),
                "confidence": conf,
                "class_name": class_name,
            }

    if best is not None:
        bbox = best["bbox"]
        if prev_bbox is not None:
            # 使用 EMA 平滑边界框，减少框抖动
            bbox = tuple(
                _TRACK_EMA * cur + (1.0 - _TRACK_EMA) * prev
                for cur, prev in zip(bbox, prev_bbox)
            )
            prev_cx, prev_cy = _bbox_center(prev_bbox)
            cur_cx, cur_cy = _bbox_center(bbox)
            vel_old_x, vel_old_y = _target_track_state["velocity"]
            vel_new = (cur_cx - prev_cx, cur_cy - prev_cy)
            velocity = (
                _TRACK_VEL_EMA * vel_old_x + (1.0 - _TRACK_VEL_EMA) * vel_new[0],
                _TRACK_VEL_EMA * vel_old_y + (1.0 - _TRACK_VEL_EMA) * vel_new[1],
            )
        else:
            velocity = (0.0, 0.0)

        _target_track_state["bbox"] = _clamp_bbox(bbox, frame_shape)
        _target_track_state["velocity"] = velocity
        _target_track_state["confidence"] = best["confidence"]
        _target_track_state["class_name"] = best["class_name"]
        _target_track_state["missed"] = 0

        return _target_from_bbox(
            _target_track_state["bbox"],
            _target_track_state["class_name"],
            _target_track_state["confidence"],
            tracked=False,
            occluded=False,
        )

    # 没检测到目标，尝试使用短时预测轨迹兜底
    if _target_track_state["bbox"] is None:
        return None

    _target_track_state["missed"] += 1
    missed = _target_track_state["missed"]
    if missed > _TRACK_MAX_MISSED:
        _target_track_state["bbox"] = None
        _target_track_state["velocity"] = (0.0, 0.0)
        _target_track_state["confidence"] = 0.0
        _target_track_state["class_name"] = None
        _target_track_state["missed"] = 0
        return None

    prev_bbox = _target_track_state["bbox"]
    vx, vy = _target_track_state["velocity"]
    damp = _TRACK_VEL_DAMP ** missed
    dx = vx * damp
    dy = vy * damp
    predicted_bbox = (prev_bbox[0] + dx, prev_bbox[1] + dy, prev_bbox[2] + dx, prev_bbox[3] + dy)
    predicted_bbox = _clamp_bbox(predicted_bbox, frame_shape)

    hand_bbox = _hand_bbox_from_results(hand_results, frame_shape)
    occluded = False
    if hand_bbox is not None:
        iou = _bbox_iou(predicted_bbox, hand_bbox)
        pcx, pcy = _bbox_center(predicted_bbox)
        hcx, hcy = _bbox_center(hand_bbox)
        dist = ((pcx - hcx) ** 2 + (pcy - hcy) ** 2) ** 0.5
        occluded = iou >= _OCCLUSION_IOU_THRESH or dist <= _OCCLUSION_CENTER_RATIO * _bbox_diag(predicted_bbox)

    _target_track_state["bbox"] = predicted_bbox
    conf = float(_target_track_state["confidence"] * (0.92 ** missed))

    return _target_from_bbox(
        predicted_bbox,
        _target_track_state["class_name"] or target_name,
        conf,
        tracked=True,
        occluded=occluded,
    )


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
# Grab detection (锁定目标 + 手部覆盖 + 深度校验)
# ---------------------------------------------------------------------------

def reset_grab_state():
    """重置抓取状态（目标切换 / 新连接时调用）。"""
    config.locked_target_bbox = None
    config.locked_target_name = None
    config.grab_state = "searching"
    config.grab_overlap_count = 0
    config.grab_no_overlap_count = 0


def _depth_close(depth_map, hand_cx, hand_cy, target_cx, target_cy,
                 tolerance: float = None) -> bool:
    """
    判断手和目标在深度方向上是否足够接近。
    如果没有深度图返回 False（不满足深度条件，防止误判）。
    """
    if depth_map is None:
        return False
    tol = tolerance if tolerance is not None else config.GRAB_DEPTH_TOLERANCE
    depth_hand = get_depth_at(depth_map, hand_cx, hand_cy)
    depth_target = get_depth_at(depth_map, target_cx, target_cy)
    if not np.isfinite(depth_hand) or not np.isfinite(depth_target):
        return False
    return abs(depth_hand - depth_target) <= tol


def update_grab_detection(target_info: dict | None,
                          hand_results,
                          frame_shape: tuple[int, int],
                          depth_map: np.ndarray | None = None) -> str:
    """
    基于锁定目标位置 + 深度校验的抓取检测状态机。

    核心思路：
    1. YOLO 检测到目标时锁定 bbox
    2. 手在 2D 上覆盖锁定区域 + 深度接近 + YOLO 丢失目标 → grabbed
    3. 仅 2D 重叠但深度不够（手在上方悬停） → 不触发抓取

    返回: "searching" | "guiding" | "close" | "grabbed"
    """
    current_target = (config.TARGET_OBJECT or "").strip().lower()

    # 目标切换时重置
    if config.locked_target_name is not None and config.locked_target_name != current_target:
        reset_grab_state()

    # 当 YOLO 直接检测到目标（非跟踪预测），更新锁定 bbox
    if target_info is not None and not target_info.get("tracked", False):
        config.locked_target_bbox = (
            target_info["x1"], target_info["y1"],
            target_info["x2"], target_info["y2"],
        )
        config.locked_target_name = current_target

    # 没有锁定位置 → 仍在搜索
    if config.locked_target_bbox is None:
        config.grab_state = "searching"
        config.grab_overlap_count = 0
        config.grab_no_overlap_count = 0
        return "searching"

    # 获取手部 bbox
    hand_bbox = _hand_bbox_from_results(hand_results, frame_shape)

    if hand_bbox is None:
        # 没有手：已抓取状态保持
        if config.grab_state == "grabbed":
            return "grabbed"
        config.grab_overlap_count = 0
        if config.grab_state == "close":
            config.grab_state = "guiding"
        return config.grab_state

    locked = config.locked_target_bbox

    # ---- 计算手与锁定目标的 2D 空间关系 ----
    iou = _bbox_iou(locked, hand_bbox)
    lcx, lcy = _bbox_center(locked)
    hcx, hcy = _bbox_center(hand_bbox)
    dist = ((lcx - hcx) ** 2 + (lcy - hcy) ** 2) ** 0.5
    diag = _bbox_diag(locked)

    # 手中心是否在锁定 bbox 内部
    hand_center_inside = (
        locked[0] <= hcx <= locked[2] and locked[1] <= hcy <= locked[3]
    )

    # 手是否在目标上方（2D 重叠）
    hand_on_target_2d = (
        iou >= 0.12
        or hand_center_inside
        or dist < diag * 0.35
    )

    # ---- 深度校验：手和目标在 Z 轴上是否接近 ----
    depth_ok = _depth_close(depth_map, hcx, hcy, lcx, lcy)

    # 目标是否被 YOLO 直接检测到
    target_visible_yolo = (
        target_info is not None and not target_info.get("tracked", False)
    )
    target_not_directly_seen = (
        target_info is None
        or target_info.get("tracked", False)
        or target_info.get("occluded", False)
    )

    logger.debug(
        f"[GRAB] iou={iou:.3f} dist={dist:.1f} diag={diag:.1f} "
        f"inside={hand_center_inside} on2d={hand_on_target_2d} "
        f"depth_ok={depth_ok} yolo_vis={target_visible_yolo} "
        f"state={config.grab_state} overlap={config.grab_overlap_count}"
    )

    # ---- 状态机 ----
    if config.grab_state == "grabbed":
        # 已抓取，检查是否释放：手远离 + 目标重新可见
        if target_visible_yolo and not hand_on_target_2d:
            config.grab_no_overlap_count += 1
            if config.grab_no_overlap_count >= config.GRAB_RELEASE_FRAMES:
                config.grab_state = "guiding"
                config.grab_overlap_count = 0
                config.grab_no_overlap_count = 0
        else:
            config.grab_no_overlap_count = 0
        return config.grab_state

    # 未抓取：检测抓取
    # 需要满足三个条件：2D 重叠 + 深度接近 + YOLO 看不到目标
    if hand_on_target_2d and depth_ok and target_not_directly_seen:
        config.grab_overlap_count += 1
        if config.grab_overlap_count >= config.GRAB_CONFIRM_FRAMES:
            config.grab_state = "grabbed"
            config.grab_no_overlap_count = 0
            logger.info("[GRAB] *** 检测到抓取成功! ***")
            return "grabbed"
        config.grab_state = "close"
        return "close"
    elif hand_on_target_2d and target_not_directly_seen:
        # 2D 重叠但深度不够（手在上方悬停），只算 close，不累加计数
        config.grab_overlap_count = 0
        config.grab_state = "close"
        return "close"
    elif hand_on_target_2d:
        # 手在目标上但目标仍可见
        config.grab_overlap_count = 0
        config.grab_state = "close"
        return "close"
    else:
        config.grab_overlap_count = 0
        config.grab_no_overlap_count = 0
        config.grab_state = "guiding"
        return "guiding"


def draw_grab_success(frame: np.ndarray, bbox: tuple) -> None:
    """绘制抓取成功的可视化效果（绿色框 + 标签）。"""
    if bbox is None:
        return
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    color = (0, 200, 0)  # 绿色

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    label = "[GRABBED]"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2,
    )
    cv2.rectangle(frame, (x1, y1 - th - baseline - 8),
                  (x1 + tw + 8, y1), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 4, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


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
