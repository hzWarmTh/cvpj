"""Model initialization: YOLOv8, MediaPipe Hands, MiDaS depth estimation."""

import logging
import functools

import torch
# Monkey-patch torch.load before any model is loaded
torch.load = functools.partial(torch.load, weights_only=False)

import mediapipe as mp
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------------
logger.info(f"强制使用计算设备: {config.DEVICE}")

yolo_model = None
try:
    # 根据 config.YOLO_MODEL_SIZE 加载对应模型（首次运行会自动下载）
    _yolo_file = f'yolov8{config.YOLO_MODEL_SIZE}.pt'
    logger.info(f"正在加载 YOLO 模型: {_yolo_file}")
    yolo_model = YOLO(_yolo_file)
    yolo_model.to('cpu')
    logger.info(f"YOLOv8 模型加载成功 ({_yolo_file})，运行设备: cpu")
except Exception as e:
    logger.error(f"YOLOv8 模型加载失败: {e}")

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
