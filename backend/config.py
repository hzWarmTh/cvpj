"""Global configuration constants and mutable runtime state."""

# --- Default target ---
TARGET_OBJECT = ""

# --- Image rotation (0, 90, 180, 270) ---
ROTATION = 0

# --- 检测到的物体列表（每帧更新） ---
detected_objects = []

# --- YOLO 模型大小 ('n'=nano极速, 's'=small平衡, 'm'=medium高精度) ---
# 'n' 最快但精度低，'s' 速度与精度兼顾（推荐），'m' 精度最高但较慢
YOLO_MODEL_SIZE = "s"

# --- 处理分辨率宽度（越小越快，推荐 320~640） ---
PROCESS_WIDTH = 480

# --- 目标输出帧率（限制 WebSocket 推送频率，避免前端积压） ---
TARGET_FPS = 30

# --- JPEG 压缩质量（1-100，越低传输越快，40 已足够清晰） ---
JPEG_QUALITY = 80

# --- Frame dimensions ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISTANCE_THRESHOLD = 50  # 像素阈值

# --- DroidCam IP 摄像头地址 ---
DROIDCAM_URL = 'http://10.11.77.167:4747/video'

# --- YOLO 帧计数 & 缓存 ---
frame_count = 0
YOLO_INTERVAL = 3          # 每 N 帧执行一次 YOLO（减小=更实时，增大=更省CPU）
yolo_cache = None           # 缓存上一次 YOLO 检测结果

# --- 深度估计间隔（MiDaS 非常耗 CPU，每 N 帧才执行一次，0=禁用） ---
DEPTH_INTERVAL = 6
depth_cache = None          # 缓存的深度图

# --- YOLO 绘制参数 (红色) ---
YOLO_BOX_COLOR = (0, 0, 255)
YOLO_TEXT_COLOR = (0, 0, 255)
YOLO_BOX_THICKNESS = 2
YOLO_FONT_SCALE = 0.6

# --- MediaPipe 手部骨骼绘制参数 (绿色) ---
HAND_LANDMARK_COLOR = (0, 255, 0)
HAND_CONNECTION_COLOR = (0, 255, 0)
HAND_LANDMARK_RADIUS = 4
HAND_CONNECTION_THICKNESS = 2

# --- 目标物体高亮绘制参数 (蓝色) ---
TARGET_BOX_COLOR = (255, 0, 0)        # 蓝色 (BGR)
TARGET_BOX_THICKNESS = 3
TARGET_FONT_SCALE = 0.7

# --- 计算设备 ---
DEVICE = 'cpu'

# --- 指令阈值 ---
COMMAND_THRESHOLD = 50  # 像素阈值
DEPTH_THRESHOLD = 0.03  # 归一化深度差阈值

# --- 指令冷却 & 防抖状态变量 ---
last_raw_instruction = ""          # 上一帧计算出的原始指令
stable_instruction_count = 0       # 当前指令连续出现的帧数
STABLE_THRESHOLD = 10              # 连续多少帧才认为稳定

last_sent_instruction = ""         # 上一次实际发送给前端的指令
last_sent_time = 0.0               # 上一次发送的时间戳
COOLDOWN_SECONDS = 3.0             # 相同指令重复发送的冷却时间（秒）

# --- 抓取检测状态 ---
locked_target_bbox = None          # 锁定的目标 bbox (x1, y1, x2, y2)，YOLO 直接检测到时更新
locked_target_name = None          # 锁定 bbox 对应的目标名称
grab_state = "searching"           # searching | guiding | close | grabbed
grab_overlap_count = 0             # 手覆盖目标区域的连续帧计数
grab_no_overlap_count = 0          # 手离开目标区域的连续帧计数（用于释放判定）
GRAB_CONFIRM_FRAMES = 4            # 确认抓取需要的连续帧数
GRAB_RELEASE_FRAMES = 10           # 确认释放需要的连续帧数
GRAB_DEPTH_TOLERANCE = 0.08        # 抓取时手和目标的深度差容忍值（归一化）
