# 视障人士视觉辅助物体抓取系统（CVPJ）

本项目是一个前后端分离的实时视觉辅助系统，目标是帮助视障用户在摄像头画面中定位目标物体，并通过语音指令引导手部靠近并抓取。

当前实现以 DroidCam 作为视频源：后端直接拉取手机摄像头视频流，执行目标检测、手部检测与空间引导，再将标注画面和文字指令通过 WebSocket 推送给前端；前端负责展示画面并用浏览器 TTS 播报指令。

## 这个项目在做什么

系统核心能力如下：

1. 识别目标物体
- 使用 YOLOv8 检测画面中的物体，并锁定指定目标（默认是 cell phone）。

2. 跟踪手部位置
- 使用 MediaPipe Hands 检测手部关键点，计算手部中心。

3. 估计前后深度关系
- 使用 MiDaS Small 做单目深度估计，判断手相对目标是更靠前还是更靠后。

4. 生成引导指令
- 按优先级先判断前后（Move Forward / Move Backward），再判断左右上下（Move Left / Move Right / Move Up / Move Down），接近后提示 Grasp。

5. 防抖与冷却
- 指令需连续稳定多帧才发送，且相同指令有冷却时间，减少语音抖动和重复轰炸。

6. 可视化与语音反馈
- 前端实时显示后端标注画面（目标框、手部骨骼等），并对新指令进行英文 TTS 播报（每条播报两遍）。

## 技术栈

### 后端
- Python 3.10+
- FastAPI + Uvicorn
- OpenCV
- MediaPipe
- Ultralytics YOLOv8
- PyTorch（用于 YOLO 与 MiDaS）
- WebSocket 实时通信

### 前端
- React 18 + Vite
- CSS
- WebSocket
- Browser SpeechSynthesis（浏览器 TTS）

## 项目结构

```text
cvpj/
├─ backend/
│  ├─ main.py               # FastAPI 服务与视觉引导主逻辑
│  ├─ requirements.txt      # Python 依赖
│  └─ yolov8n.pt            # YOLO 模型权重
├─ frontend/
│  ├─ package.json
│  ├─ vite.config.js
│  ├─ src/
│  │  ├─ App.jsx            # 前端主界面与 WebSocket/TTS 逻辑
│  │  ├─ App.css
│  │  ├─ main.jsx
│  │  └─ index.css
│  └─ public/
├─ pj_req.md                # 项目需求说明
└─ README.md
```

## 后端接口概览

1. GET /
- 健康检查。

2. POST /set-target
- 设置全局目标物体。
- 请求示例：

```json
{
  "target": "cup"
}
```

3. WebSocket /ws/video（前端当前在用）
- 后端主动从 DroidCam 拉流。
- 返回：

```json
{
  "image": "data:image/jpeg;base64,...",
  "instruction": "Move Left",
  "display_instruction": "Move Left",
  "speech_instruction": "Move Left"
}
```

字段说明：
- display_instruction：每帧返回，用于前端持续显示当前指令。
- speech_instruction：仅在稳定且通过冷却后返回，用于语音播报。

4. WebSocket /ws/guidance（保留接口）
- 客户端上传 frame，后端返回检测结果与指令。

## 数据流（当前实际运行路径）

1. 前端连接 ws://localhost:8000/ws/video。
2. 后端从 main.py 中配置的 DroidCam 地址读取视频帧。
3. 后端进行 YOLO + MediaPipe + MiDaS 推理。
4. 后端输出带标注的图像和引导指令。
5. 前端渲染图像，并对新指令进行 TTS 播报。

## 快速开始

### 1) 后端

在项目根目录执行：

```bash
cd backend
pip install -r requirements.txt
python main.py
```

启动后地址：
- http://localhost:8000
- 文档：http://localhost:8000/docs

重要说明：
- 建议在 backend 目录下启动，否则相对路径模型文件 yolov8n.pt 可能找不到。

### 2) 前端

新开终端，在项目根目录执行：

```bash
cd frontend
npm install
npm run dev
```

Vite 默认地址通常是：
- http://localhost:5173

## 运行前配置

请先在 backend/main.py 中确认以下配置：

1. DroidCam 视频流地址
- 变量 DROIDCAM_URL
- 例如：http://你的手机IP:4747/video

2. 目标类别
- 默认 TARGET_OBJECT 为 cell phone
- 可通过 POST /set-target 动态修改

3. 指令灵敏度
- COMMAND_THRESHOLD、DEPTH_THRESHOLD、STABLE_THRESHOLD、COOLDOWN_SECONDS 可按场景微调

## 当前实现与需求文档的关系

- 已实现：目标检测、手部检测、实时回传、语音引导、目标切换接口。
- 当前差异：前端主流程不是上传本地摄像头帧，而是连接 ws/video 接收后端拉取的 DroidCam 画面。
- 这意味着：网页端不依赖 react-webcam 才能工作，但必须保证手机 DroidCam 可访问。

## 常见问题排查

1. 页面没有画面
- 检查 DroidCam 是否启动，手机与电脑网络是否互通。
- 检查 DROIDCAM_URL 是否可在浏览器直接访问。

2. 前端提示 WebSocket 错误
- 确认后端已启动在 8000 端口。
- 检查前端是否仍连接 ws://localhost:8000/ws/video。

3. 指令频繁抖动
- 增大 STABLE_THRESHOLD。
- 增大 COMMAND_THRESHOLD 或 DEPTH_THRESHOLD。

4. 推理速度慢
- 当前已固定使用 CPU。
- 可降低输入分辨率、调大 YOLO_INTERVAL、或改用更高性能设备。

## 依赖

- 后端依赖见 backend/requirements.txt
- 前端依赖见 frontend/package.json

## 备注

本项目用于课程实践与技术演示。
