"""Guidance instruction generation and debounce/cooldown logic."""

import logging
import time

import numpy as np

import config
from vision import get_depth_at

logger = logging.getLogger(__name__)


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
    if target is None:
        return f"Target {target_name} not found"
    if hand is None:
        return "Hand not found"

    # ---------- 第一层：Z轴/深度判断（优先） ----------
    if depth_map is not None:
        depth_hand = get_depth_at(depth_map, hand["cx"], hand["cy"])
        depth_obj = get_depth_at(depth_map, target["cx"], target["cy"])
        diff = depth_hand - depth_obj
        print(f"DEBUG: Hand Depth={depth_hand:.2f}, Obj Depth={depth_obj:.2f}, Diff={diff:.2f}")
        logger.debug(f"[DEPTH] hand={depth_hand:.3f} obj={depth_obj:.3f} diff={diff:.3f}")

        if diff > depth_threshold:
            print(f"Logic: Hand is farther (depth_hand={depth_hand:.2f} > depth_obj={depth_obj:.2f}), returning 'Move Forward'")
            return "Move Forward"
        elif diff < -depth_threshold:
            print(f"Logic: Hand is closer (depth_hand={depth_hand:.2f} < depth_obj={depth_obj:.2f}), returning 'Move Backward'")
            return "Move Backward"

    # ---------- 第二层：X/Y轴对齐 ----------
    dx = target["cx"] - hand["cx"]
    dy = target["cy"] - hand["cy"]

    if abs(dx) <= threshold and abs(dy) <= threshold:
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
