"""语音处理模块：VAD + Whisper ASR + 意图解析"""

import logging
import io
import numpy as np
import wave

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 简单能量 VAD（无需下载模型，本地即可运行）
# ---------------------------------------------------------------------------

class SimpleVAD:
    """
    基于能量的简单 VAD 实现
    不需要外部模型下载，完全本地运行
    """
    
    def __init__(self, sample_rate: int = 16000, 
                 energy_threshold: float = 0.01,
                 min_speech_frames: int = 3,
                 min_silence_frames: int = 15):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        
        # 状态
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.max_speech_duration = 10  # 秒
    
    def reset(self):
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """计算音频能量（RMS）"""
        return np.sqrt(np.mean(audio ** 2))
    
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """处理音频块"""
        energy = self._compute_energy(audio_chunk)
        is_speech = bool(energy > self.energy_threshold)  # 转为 Python bool
        
        result = {
            'is_speech': is_speech,
            'speech_started': False,
            'speech_ended': False,
            'audio_data': None,
            'speech_prob': float(min(energy / (self.energy_threshold * 2), 1.0))  # 转为 Python float
        }
        
        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
            self.speech_buffer.append(audio_chunk)
            
            if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                result['speech_started'] = True
                logger.info("VAD: 检测到语音开始")
            
            # 检查最大时长
            buffer_duration = len(self.speech_buffer) * len(audio_chunk) / self.sample_rate
            if buffer_duration > self.max_speech_duration:
                result['speech_ended'] = True
                result['audio_data'] = self._get_audio_bytes()
                self.reset()
        else:
            self.speech_frames = 0
            
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_buffer.append(audio_chunk)
                
                if self.silence_frames >= self.min_silence_frames:
                    result['speech_ended'] = True
                    result['audio_data'] = self._get_audio_bytes()
                    logger.info("VAD: 检测到语音结束")
                    self.reset()
        
        return result
    
    def _get_audio_bytes(self) -> bytes:
        if not self.speech_buffer:
            return None
        
        audio_data = np.concatenate(self.speech_buffer)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()


# ---------------------------------------------------------------------------
# VAD 处理器（使用简单能量 VAD）
# ---------------------------------------------------------------------------

class VADProcessor:
    """
    语音活动检测处理器
    使用简单能量检测，无需外部模型
    """
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.02):
        self.vad = SimpleVAD(
            sample_rate=sample_rate,
            energy_threshold=threshold,
            min_speech_frames=2,
            min_silence_frames=12
        )
    
    def reset(self):
        self.vad.reset()
    
    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        return self.vad.process_chunk(audio_chunk)


# ---------------------------------------------------------------------------
# Whisper ASR 模型
# ---------------------------------------------------------------------------

_whisper_model = None
WHISPER_MODEL_SIZE = "tiny"  # tiny/base/small - 越小越快

def load_whisper_model():
    """加载 Whisper 模型（首次调用时下载）"""
    global _whisper_model
    if _whisper_model is None:
        logger.info(f"正在加载 Whisper {WHISPER_MODEL_SIZE} 模型...")
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info(f"Whisper {WHISPER_MODEL_SIZE} 模型加载完成")
    return _whisper_model


def get_whisper_model():
    """获取已加载的 Whisper 模型"""
    if _whisper_model is None:
        load_whisper_model()
    return _whisper_model


# ---------------------------------------------------------------------------
# 语音识别
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    使用 Whisper 将音频转为文字
    
    Args:
        audio_bytes: WAV 格式音频数据
    
    Returns:
        str: 识别的文字
    """
    import tempfile
    import os
    
    model = get_whisper_model()
    
    # 写入临时文件（Whisper 需要文件路径）
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        # 执行转录
        result = model.transcribe(
            temp_path,
            language='en',  # 限定英语以提高准确率
            fp16=False,     # CPU 使用 fp32
        )
        text = result.get('text', '').strip()
        logger.info(f"Whisper 识别结果: {text}")
        return text
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# 意图解析
# ---------------------------------------------------------------------------

# 唤醒词列表（仅匹配 hi/hey/hello tom 及常见 Whisper 误识别）
WAKE_WORDS = [
    "hi tom", "hey tom", "hello tom",
    # Whisper 常见误识别变体
    "hi time", "hey time", "hello time",
    "hi tim", "hey tim",
    "hi tam", "hey tam",
    "hi tong", "hey tong",
]

# 目标查询关键词（支持 "where are the [object]" 和 "where is the [object]"）
TARGET_QUERY_KEYWORDS = [
    "where are the", "where are", "where is the", "where is", "where's",
]

# 抓取查询关键词（仅支持 "did i get it"）
GRASP_QUERY_KEYWORDS = [
    "did i get it", "did i get"
]


class IntentResult:
    """意图解析结果"""
    
    # 意图类型
    WAKE = "wake"                     # 唤醒
    SELECT_TARGET = "select_target"   # 选择目标
    QUERY_LOCATION = "query_location" # 查询位置
    QUERY_GRASP = "query_grasp"       # 查询是否抓到
    STOP = "stop"                     # 停止
    UNKNOWN = "unknown"               # 未知
    
    def __init__(self, intent_type: str, target: str = None, raw_text: str = ""):
        self.intent_type = intent_type
        self.target = target
        self.raw_text = raw_text
    
    def __repr__(self):
        return f"IntentResult(type={self.intent_type}, target={self.target})"


def parse_intent(text: str, detected_objects: list = None, is_awake: bool = False) -> IntentResult:
    """
    解析语音文字的意图
    
    Args:
        text: 识别的文字
        detected_objects: 当前检测到的物体列表
        is_awake: 当前是否已唤醒
    
    Returns:
        IntentResult: 解析结果
    """
    if not text:
        logger.info("意图解析: 空文本")
        return IntentResult(IntentResult.UNKNOWN, raw_text=text)
    
    text_lower = text.lower().strip()
    # 移除标点符号
    text_clean = ''.join(c for c in text_lower if c.isalnum() or c.isspace())
    
    logger.info(f"意图解析: 原文='{text}', 清理后='{text_clean}', 已唤醒={is_awake}")
    
    # 处理 detected_objects（可能是 dict 列表或字符串列表）
    detected_objects = detected_objects or []
    object_names = []
    for obj in detected_objects:
        if isinstance(obj, dict):
            object_names.append(obj.get('name', ''))
        else:
            object_names.append(str(obj))
    detected_lower = [name.lower() for name in object_names if name]
    
    # 1. 检查唤醒词（始终检查，无论是否已唤醒）
    for wake_word in WAKE_WORDS:
        if wake_word in text_clean or wake_word in text_lower:
            logger.info(f"意图解析: 匹配唤醒词 '{wake_word}'")
            return IntentResult(IntentResult.WAKE, raw_text=text)
    
    # 如果未唤醒，忽略其他指令
    if not is_awake:
        logger.info("意图解析: 未唤醒，忽略非唤醒指令")
        return IntentResult(IntentResult.UNKNOWN, raw_text=text)

    # 2. 检查抓取查询："did i get it?"
    for keyword in GRASP_QUERY_KEYWORDS:
        if keyword in text_lower:
            return IntentResult(IntentResult.QUERY_GRASP, raw_text=text)

    # 3. 检查位置查询（按长度降序匹配，优先匹配最长关键词）
    # 例如："where are the bottle"
    for keyword in sorted(TARGET_QUERY_KEYWORDS, key=len, reverse=True):
        if keyword in text_lower:
            target = _extract_target_from_query(text_lower, keyword, detected_lower, object_names)
            return IntentResult(IntentResult.QUERY_LOCATION, target=target, raw_text=text)

    logger.info("未识别到有效命令")
    return IntentResult(IntentResult.UNKNOWN, raw_text=text)


def _extract_target_from_query(text: str, keyword: str, detected_lower: list, detected_objects: list) -> str:
    """从查询语句中提取目标物体名称"""
    # 找到关键词后面的部分
    idx = text.find(keyword)
    if idx >= 0:
        after_keyword = text[idx + len(keyword):].strip()
        # 移除常见词
        for word in ["the", "my", "a", "an"]:
            if after_keyword.startswith(word + " "):
                after_keyword = after_keyword[len(word) + 1:]
        
        # 在检测到的物体中匹配
        for i, obj_lower in enumerate(detected_lower):
            if obj_lower in after_keyword:
                return detected_objects[i]
        
        # 返回提取的文字
        words = after_keyword.split()
        if words:
            return words[0]
    
    return None


# ---------------------------------------------------------------------------
# 预加载模型
# ---------------------------------------------------------------------------

def preload_models():
    """预加载 Whisper 模型（可在后台调用）"""
    logger.info("开始预加载语音模型...")
    load_whisper_model()
    logger.info("语音模型预加载完成")
