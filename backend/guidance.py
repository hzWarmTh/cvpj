"""Guidance instruction generation and debounce/cooldown logic."""

import logging
import time

import numpy as np

import config
from vision import get_depth_at

logger = logging.getLogger(__name__)


# 深度方向状态：抑制 Forward/Backward 在临界深度附近来回跳变
_depth_diff_ema = 0.0
_depth_direction = 0  # 1=Forward, -1=Backward, 0=XY
_DEPTH_EMA_ALPHA = 0.25
_DEPTH_ENTER_ABS = 0.05
_DEPTH_RELEASE_ABS = 0.03
_DEPTH_SWITCH_ABS = 0.07
_DEPTH_GRASP_ABS = 0.025


# ---------------------------------------------------------------------------
# Instruction generators
# ---------------------------------------------------------------------------

def generate_command(target: dict | None, hand: dict | None,
                     threshold: int = config.COMMAND_THRESHOLD) -> dict:
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
                         target_name: str = config.TARGET_OBJECT,
                         threshold: int = config.COMMAND_THRESHOLD,
                         depth_map: np.ndarray = None,
                         depth_threshold: float = config.DEPTH_THRESHOLD) -> str:
    """
    根据目标与手部的空间关系生成指令字符串。
    优先级：Z轴（前后） > X/Y轴（左右上下）
    """
    global _depth_diff_ema, _depth_direction

    if target is None:
        _depth_direction = 0
        return f"Target {target_name} not found"
    if hand is None:
        _depth_direction = 0
        return "Hand not found"

    depth_ema_abs = None
    depth_ema_sign = 0

    # ---------- 第一层：Z轴/深度判断（优先） ----------
    if depth_map is not None:
        depth_hand = get_depth_at(depth_map, hand["cx"], hand["cy"])
        depth_obj = get_depth_at(depth_map, target["cx"], target["cy"])
        if np.isfinite(depth_hand) and np.isfinite(depth_obj):
            diff = depth_hand - depth_obj
            _depth_diff_ema = _DEPTH_EMA_ALPHA * diff + (1.0 - _DEPTH_EMA_ALPHA) * _depth_diff_ema
            depth_ema_abs = abs(_depth_diff_ema)
            depth_ema_sign = 1 if _depth_diff_ema > 0 else (-1 if _depth_diff_ema < 0 else 0)

            # 只有明显深度差才进入 Z 轴指令；深度接近时快速释放回 XY
            z_enter = max(_DEPTH_ENTER_ABS, depth_threshold * 1.8)
            z_release = max(_DEPTH_RELEASE_ABS, depth_threshold * 1.2)
            z_switch = max(_DEPTH_SWITCH_ABS, depth_threshold * 2.3)
            abs_ema = depth_ema_abs

            if _depth_direction == 0:
                if abs_ema >= z_enter:
                    # 第一人称语义：手更靠近摄像头(更大) -> 前伸；手更远 -> 回拉
                    _depth_direction = 1 if _depth_diff_ema > 0 else -1
            else:
                if abs_ema <= z_release:
                    _depth_direction = 0
                elif _depth_direction == 1 and _depth_diff_ema < -z_switch:
                    _depth_direction = -1
                elif _depth_direction == -1 and _depth_diff_ema > z_switch:
                    _depth_direction = 1

            logger.debug(
                f"[DEPTH] hand={depth_hand:.3f} obj={depth_obj:.3f} "
                f"diff={diff:.3f} ema={_depth_diff_ema:.3f} "
                f"enter={z_enter:.3f} release={z_release:.3f} "
                f"switch={z_switch:.3f} dir={_depth_direction}"
            )

            if _depth_direction == 1:
                return "Move Forward"
            elif _depth_direction == -1:
                return "Move Backward"

    # ---------- 第二层：X/Y轴对齐 ----------
    dx = target["cx"] - hand["cx"]
    dy = target["cy"] - hand["cy"]

    if abs(dx) <= threshold and abs(dy) <= threshold:
        # XY 已对齐时，若深度仍未对齐则继续给前后指令，避免过早 Grasp
        if depth_ema_abs is not None:
            z_grasp = max(_DEPTH_GRASP_ABS, depth_threshold * 0.8)
            if depth_ema_abs > z_grasp:
                return "Move Forward" if depth_ema_sign > 0 else "Move Backward"
        return "Grasp"

    if abs(dx) >= abs(dy):
        if dx > threshold:
            return "Move Right"
        elif dx < -threshold:
            return "Move Left"

    if dy > threshold:
        return "Move Down"
    elif dy < -threshold:
        return "Move Up"

    return "Grasp"


def generate_guidance_command(target_info: dict, hand_info: dict) -> dict:
    """生成语音指导指令（用于 /ws/guidance 端点）"""
    if not target_info or not hand_info.get("detected"):
        return {
            "command": "wait",
            "message": "正在等待检测目标物体和手部...",
        }

    target_x = target_info["center_x"]
    target_y = target_info["center_y"]
    hand_x = hand_info["center_x"]
    hand_y = hand_info["center_y"]

    dx = hand_x - target_x
    dy = hand_y - target_y
    distance = (dx ** 2 + dy ** 2) ** 0.5

    if distance < config.DISTANCE_THRESHOLD:
        return {
            "command": "grasp",
            "message": "距离足够，现在可以抓取了！",
            "distance": float(distance),
        }

    if abs(dx) > abs(dy):
        if dx > 0:
            return {
                "command": "move_left",
                "message": "请向左移动你的手。",
                "distance": float(distance),
            }
        else:
            return {
                "command": "move_right",
                "message": "请向右移动你的手。",
                "distance": float(distance),
            }
    else:
        if dy > 0:
            return {
                "command": "move_up",
                "message": "请向上移动你的手。",
                "distance": float(distance),
            }
        else:
            return {
                "command": "move_down",
                "message": "请向下移动你的手。",
                "distance": float(distance),
            }


# ---------------------------------------------------------------------------
# Debounce & cooldown
# ---------------------------------------------------------------------------

def stabilize_instruction(raw_instruction: str) -> str:
    """
    对指令进行防抖 + 冷却过滤。
    返回值为需要语音播报的指令（空字符串 = 不播报）。
    """
    current_time = time.time()

    if raw_instruction == config.last_raw_instruction:
        config.stable_instruction_count += 1
    else:
        config.stable_instruction_count = 1
        config.last_raw_instruction = raw_instruction

    instruction_to_send = ""

    if config.stable_instruction_count >= config.STABLE_THRESHOLD:
        if raw_instruction != config.last_sent_instruction:
            # 指令变了，立即发送
            instruction_to_send = raw_instruction
            config.last_sent_instruction = raw_instruction
            config.last_sent_time = current_time
        else:
            # 指令未变，检查冷却时间
            if current_time - config.last_sent_time > config.COOLDOWN_SECONDS:
                instruction_to_send = raw_instruction
                config.last_sent_time = current_time

    return instruction_to_send
