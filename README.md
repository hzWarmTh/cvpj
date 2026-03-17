# 视障人士视觉辅助系统 (Visual Assistance System)

基于计算机视觉和深度学习的实时视觉辅助系统，帮助视障人士通过语音指令引导手部准确抓取目标物体。系统通过手机摄像头（DroidCam）实时采集画面，利用 YOLOv8 物体检测、MediaPipe 手部追踪和 MiDaS 单目深度估计，生成六方向空间语音导航指令（上/下/左/右/前/后），引导用户完成抓取动作。

---

## 目录

- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [启动运行](#启动运行)
- [使用说明](#使用说明)
- [配置参数](#配置参数)
- [API 接口](#api-接口)
- [常见问题](#常见问题)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         整体架构                                 │
│                                                                 │
│  ┌──────────┐    HTTP/MJPEG     ┌──────────────────────────┐   │
│  │ DroidCam │ ───────────────── │     Python 后端           │   │
│  │ (手机)    │    视频流         │     FastAPI + uvicorn     │   │
│  └──────────┘                   │                          │   │
│                                 │  ┌─────────────────────┐ │   │
│                                 │  │ FrameGrabber        │ │   │
│                                 │  │ (后台线程实时拉帧)    │ │   │
│                                 │  └────────┬────────────┘ │   │
│                                 │           │              │   │
│                                 │  ┌────────▼────────────┐ │   │
│                                 │  │ 推理管线             │ │   │
│                                 │  │ YOLOv8s 物体检测     │ │   │
│                                 │  │ MediaPipe 手部追踪   │ │   │
│                                 │  │ MiDaS 深度估计       │ │   │
│                                 │  └────────┬────────────┘ │   │
│                                 │           │              │   │
│                                 │  ┌────────▼────────────┐ │   │
│                                 │  │ 指令生成器           │ │   │
│                                 │  │ 六方向+抓取 指令     │ │   │
│                                 │  │ 防抖 + 冷却过滤      │ │   │
│                                 │  └────────┬────────────┘ │   │
│                                 └───────────┼──────────────┘   │
│                                    WebSocket │                  │
│                                 ┌───────────▼──────────────┐   │
│                                 │    React 前端             │   │
│                                 │    实时画面 + 指令显示     │   │
│                                 │    物体选择 + 画面旋转     │   │
│                                 │    TTS 语音播报           │   │
│                                 └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
DroidCam (手机摄像头)
    │
    │  MJPEG over HTTP
    ▼
FrameGrabber (后台线程持续抓取最新帧)
    │
    │  最新帧
    ▼
_process_frame_sync (ThreadPoolExecutor 单线程推理)
    ├── 旋转 & 缩放
    ├── YOLOv8s 物体检测 (每3帧/缓存)
    ├── MediaPipe Hands 手部追踪 (每帧)
    ├── MiDaS 深度估计 (每15帧/缓存)
    ├── 指令生成 (Z > X/Y 优先级)
    ├── 防抖 + 冷却过滤
    └── JPEG 编码
    │
    │  WebSocket JSON
    ▼
React 前端
    ├── 实时画面渲染
    ├── 指令文字显示
    ├── TTS 语音播报 (每条读两遍)
    └── 用户交互 (目标选择/画面旋转)
```

---

## 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.10 | 运行环境 |
| **FastAPI** | 0.104.1 | Web 框架，HTTP + WebSocket |
| **uvicorn** | 0.24.0 (standard) | ASGI 服务器，WebSocket 支持 |
| **YOLOv8s** | ultralytics 8.0.224 | 实时物体检测（80类 COCO 物体） |
| **MediaPipe** | 0.10.9 | 手部 21 关键点追踪 |
| **MiDaS Small** | intel-isl/MiDaS | 单目深度估计（前后方向判断） |
| **PyTorch** | 2.1.1 | 深度学习推理框架 |
| **OpenCV** | 4.8.1.78 | 图像处理、视频流读取 |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| **React** | 18.2 | UI 框架 |
| **Vite** | 5.x | 构建工具，开发服务器 |
| **Web Speech API** | 浏览器内置 | TTS 语音合成播报 |
| **WebSocket** | 浏览器内置 | 与后端实时通信 |
| **Tailwind CSS** | 3.3 | 样式工具（辅助） |

### 外部设备

| 设备 | 用途 |
|------|------|
| **DroidCam** | 手机充当 IP 摄像头，提供 MJPEG 视频流 |

---

## 项目结构

```
cvpj/
├── README.md                    # 本文件 - 项目说明
├── 技术实现说明.md               # 详细技术实现文档
├── pj_req.md                    # 原始项目需求
│
├── backend/                     # Python 后端
│   ├── main.py                  # 应用入口 (FastAPI + uvicorn 启动)
│   ├── config.py                # 全局配置常量 & 运行时状态
│   ├── models.py                # 模型加载 (YOLO / MediaPipe / MiDaS)
│   ├── vision.py                # 视觉工具 (帧处理 / 绘制 / 检测 / 深度)
│   ├── guidance.py              # 指令生成 & 防抖冷却逻辑
│   ├── routes.py                # HTTP + WebSocket 路由处理
│   └── requirements.txt         # Python 依赖列表
│
└── frontend/                    # React 前端
    ├── index.html               # HTML 入口
    ├── package.json             # npm 依赖配置
    ├── vite.config.js           # Vite 构建配置 (端口 3000, 代理)
    ├── tailwind.config.js       # Tailwind 配置
    ├── postcss.config.js        # PostCSS 配置
    └── src/
        ├── main.jsx             # React 入口
        ├── App.jsx              # 主组件 (视频/控制面板/TTS)
        ├── App.css              # 全局样式
        └── index.css            # 基础样式
```

---

## 环境要求

- **操作系统**: Windows 10/11（已测试）、macOS、Linux
- **Python**: 3.10（推荐通过 Anaconda/Miniconda 管理）
- **Node.js**: 16+（前端构建）
- **手机**: 安装 DroidCam 应用，与电脑处于同一局域网
- **浏览器**: Chrome / Edge（需支持 Web Speech API）

> **硬件建议**: 推理全部运行在 CPU 上，建议 8 核以上处理器以获得流畅体验。如有 NVIDIA GPU，可修改 `config.py` 中 `DEVICE = 'cuda'` 以加速。

---

## 安装部署

### 1. 克隆项目

```bash
git clone <仓库地址>
cd cvpj
```

### 2. 创建 Conda 环境并安装后端依赖

```bash
# 创建 Python 3.10 环境
conda create -n cvpj python=3.10 -y
conda activate cvpj

# 安装后端依赖
cd backend
pip install -r requirements.txt
```

> **首次运行时**，YOLOv8s 模型权重 (`yolov8s.pt`, ~21.5MB) 和 MiDaS Small 模型会由框架自动下载到本地缓存，无需手动处理。

### 3. 安装前端依赖

```bash
cd ../frontend
npm install
```

### 4. 配置 DroidCam 地址

编辑 `backend/config.py`，将 `DROIDCAM_URL` 修改为你的手机 DroidCam 地址：

```python
DROIDCAM_URL = 'http://<手机IP>:<端口>/video'
# 示例: DROIDCAM_URL = 'http://192.168.1.100:4747/video'
```

获取方法：打开手机 DroidCam 应用，界面上会显示 IP 和端口。

---

## 启动运行

### 启动后端

```bash
conda activate cvpj
cd backend
python main.py
```

看到以下输出即表示启动成功：
```
INFO:     YOLOv8 模型加载成功 (yolov8s.pt)，运行设备: cpu
INFO:     MiDaS Small 深度估计模型加载成功 (CPU)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 启动前端

新开一个终端窗口：

```bash
cd frontend
npm run dev
```

看到以下输出：
```
VITE v5.x.x  ready in xxx ms
➜  Local:   http://localhost:3000/
```

### 打开浏览器

访问 **http://localhost:3000**，即可看到系统界面。

---

## 使用说明

### 基本操作流程

1. **确保手机 DroidCam 已开启**，并确认手机与电脑在同一局域网
2. **启动后端**（`python main.py`），等待模型加载完成
3. **启动前端**（`npm run dev`），打开浏览器访问 `http://localhost:3000`
4. **点击「Start」按钮**，系统开始实时检测
5. **画面中识别到的物体**会以可点击按钮形式显示在右侧控制面板
6. **点击物体名称**（如 `cup`、`bottle`），锁定为抓取目标
7. 系统会自动判断手部与目标的空间关系，生成并播报语音指令
8. **根据语音提示移动手部**：Move Left / Right / Up / Down / Forward / Backward
9. 当手部与目标足够接近时，系统提示 **"Grasp"**（抓取）

### 画面旋转

如果摄像头画面方向不对（上下颠倒或侧翻）：
- 点击视频区域左上角的 **旋转按钮**
- 每次点击旋转 90°，循环：0° → 90° → 180° → 270° → 0°

### 目标物体选择

- **自动检测**：系统使用 YOLOv8 自动识别画面中所有物体，以按钮芯片形式展示
- **一键切换**：点击任意物体名称即可将其设为目标，无需手动输入
- **实时更新**：物体列表随画面内容实时刷新
- YOLOv8 支持 COCO 数据集的 **80 类常见物体**（杯子、手机、瓶子、书、遥控器等）

### 语音指令说明

| 指令 | 含义 |
|------|------|
| `Move Left` | 手需要向左移动 |
| `Move Right` | 手需要向右移动 |
| `Move Up` | 手需要向上移动 |
| `Move Down` | 手需要向下移动 |
| `Move Forward` | 手需要向前伸（靠近物体） |
| `Move Backward` | 手需要向后缩（远离物体） |
| `Grasp` | 手已到达目标位置，可以抓取 |
| `Target xxx not found` | 画面中未检测到目标物体 |
| `Hand not found` | 画面中未检测到手部 |

> 语音指令采用防抖 + 冷却机制：连续 10 帧稳定后才播报，相同指令 3 秒内不重复播报，避免语音轰炸。每条指令自动读两遍，确保用户听清。

---

## 配置参数

所有可调参数集中在 `backend/config.py` 中：

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DROIDCAM_URL` | `http://10.11.68.30:4747/video` | DroidCam 摄像头地址 |
| `TARGET_OBJECT` | `"cell phone"` | 默认目标物体（运行时可通过前端切换） |
| `YOLO_MODEL_SIZE` | `"s"` | YOLO 模型大小：`n`(nano极速) / `s`(推荐平衡) / `m`(高精度) |

### 性能调优

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PROCESS_WIDTH` | `480` | 处理分辨率宽度（越小越快，范围 320~640） |
| `TARGET_FPS` | `15` | WebSocket 推送帧率上限 |
| `JPEG_QUALITY` | `40` | JPEG 压缩质量（1-100，40 已足够清晰） |
| `YOLO_INTERVAL` | `3` | 每 N 帧执行一次 YOLO 推理 |
| `DEPTH_INTERVAL` | `15` | 每 N 帧执行一次深度估计（0=禁用） |

### 指令控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `COMMAND_THRESHOLD` | `50` | 手与目标距离阈值（像素），小于此值判定为"Grasp" |
| `DEPTH_THRESHOLD` | `0.03` | 归一化深度差阈值，超过此值才生成前后指令 |
| `STABLE_THRESHOLD` | `10` | 连续多少帧相同才认为指令稳定 |
| `COOLDOWN_SECONDS` | `3.0` | 相同指令重复播报的冷却时间（秒） |

---

## API 接口

### HTTP 接口

| 方法 | 路径 | 请求体 | 说明 |
|------|------|--------|------|
| `GET` | `/` | 无 | 健康检查，返回状态信息 |
| `POST` | `/set-target` | `{"target": "cup"}` | 设置目标物体 |
| `POST` | `/set-rotation` | `{"rotation": 90}` | 设置画面旋转角度（0/90/180/270） |

### WebSocket 接口

| 路径 | 方向 | 说明 |
|------|------|------|
| `/ws/video` | 后端→前端 | 主端点：后端拉取 DroidCam → 推理 → 推送标注画面 + 指令 |
| `/ws/video` | 前端→后端 | 前端发送目标切换/旋转设置 |
| `/ws/guidance` | 双向 | 备用端点：前端上传帧 → 后端处理 → 返回指令 |

#### `/ws/video` 后端推送 JSON 格式

```json
{
  "image": "data:image/jpeg;base64,...",
  "instruction": "Move Left",
  "display_instruction": "Move Left",
  "speech_instruction": "Move Left",
  "detected_objects": ["cup", "bottle", "cell phone"],
  "target": "cup"
}
```

其中 `speech_instruction` 经过防抖+冷却过滤，为空时前端不播报。

#### `/ws/video` 前端可发送的 JSON 消息

```json
{"target": "bottle"}
{"rotation": 180}
```

---

## 常见问题

### Q: 启动后端报错 websockets 相关错误

安装 uvicorn 的标准版本（含 WebSocket 支持）：
```bash
pip install "uvicorn[standard]==0.24.0"
```
> 注意：`websockets` 版本必须 < 13.0。`uvicorn[standard]==0.24.0` 会自动安装兼容版本。

### Q: DroidCam 无法连接

1. 确认手机和电脑在同一局域网（同一 WiFi）
2. 确认手机 DroidCam 应用已启动
3. 在电脑浏览器中直接访问 `http://<手机IP>:4747/video`，能看到视频流即正常
4. 检查 `config.py` 中的 `DROIDCAM_URL` 是否正确

### Q: 画面有延迟

系统已内置多层延迟优化（FrameGrabber 后台拉帧 + 推理线程池 + 帧率限制）。如果仍有延迟：
- 降低 `PROCESS_WIDTH`（如改为 320）
- 增大 `YOLO_INTERVAL`（如改为 5）
- 设置 `DEPTH_INTERVAL = 0` 禁用深度估计
- 降低 `JPEG_QUALITY`（如改为 30）

### Q: 物体检测精度不够

- 修改 `config.py` 中 `YOLO_MODEL_SIZE = "m"`（medium 模型精度最高，mAP 50.2，但较慢）
- 确保光线充足，目标物体完整出现在画面中
- YOLOv8 支持 COCO 数据集 80 类物体，输入 `cup`、`bottle`、`cell phone` 等英文名

### Q: MiDaS 模型加载失败

```bash
pip install timm>=0.9.0
```
MiDaS 依赖 `timm`（PyTorch Image Models）库。

### Q: 前端 npm install 失败

确保 Node.js >= 16：
```bash
node --version
npm --version
```

### Q: PowerShell 中 npm 命令报执行策略错误

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

---

## 许可证

本项目仅用于学术研究和辅助功能开发。
