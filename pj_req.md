COMP5523 期末项目：视障人士视觉辅助物体抓取系统 - 技术规范

1. 项目概述

开发一个 B/S 架构的视觉辅助系统，帮助视障用户抓取物体。用户通过网页前端开启摄像头和麦克风，说出目标物体名称（如 "bottle"）。后端实时分析视频流，识别目标与手部，并通过 WebSocket 返回实时的视觉标注画面和音频指导指令，前端播放语音引导用户抓取。

2. 技术栈锁定

后端
语言: Python 3.10+
Web框架: FastAPI (用于 WebSocket 实时通信)
计算机视觉:
OpenCV: 图像编解码、基础处理。
MediaPipe: 手部关键点检测 (Hand Tracking)。
Ultralytics (YOLOv8): 通用物体检测 (Object Detection)。
语音识别: SpeechRecognition 库 (或模拟接收文本指令)。

前端
框架: React (使用 Vite 构建工具)
样式: Tailwind CSS (快速构建美观 UI)
核心功能:
react-webcam: 摄像头采集。
WebSocket Client: 实时数据传输。
Browser TTS: 浏览器原生语音合成播放指令。
通信协议
WebSocket (/ws/guidance): 全双工通信。
Client -> Server: 发送视频帧 (Base64 图片)。
Server -> Client: 返回处理后图片 + JSON 格式指导指令。

3. 功能模块详细设计

3.1 目标检测与手部追踪
检测流程:
接收前端视频帧。
YOLOv8 检测画面中所有物体，筛选出用户指定的目标物体，记录其中心坐标。
MediaPipe 检测手部 21 个关键点，计算手部中心坐标及状态 (张开/握拳)。
性能要求: 需处理至可接受的帧率 (10-15 FPS)。

3.2 空间估算与指导逻辑
坐标系定义: 以图像中心为原点，X轴控制左右，Y轴控制上下。
指令生成规则:
dx = hand_center_x - target_center_x
dy = hand_center_y - target_center_y
根据 dx 和 dy 的阈值，生成 "Move Left/Right/Up/Down"。
若距离小于阈值，生成 "Stop" 和 "Grasp" (抓取) 指令。

3.3 前端交互界面
界面布局:
左侧: 大面积视频展示区 (显示摄像头画面及后端回传的标注框)。
右侧: 控制面板 (显示当前状态、目标物体名称、开始/停止按钮)。
音频交互:
用户点击“开始”后，前端持续监听语音或输入框文本。
收到后端指令文本后，自动调用 speechSynthesis 播放。

4. 数据流设计

初始化: 用户访问网页，允许摄像头权限。
设定目标: 用户输入 "cup" -> 发送给后端 -> 后端切换检测目标。
循环引导:
前端每 100ms 截取一帧图片 -> 转Base64 -> WebSocket发送。
后端接收 -> 解码 -> YOLO+MediaPipe推理 -> 计算相对位置 -> 编码标注图。
后端返回 JSON { "image": "base64...", "command": "move_right", "message": "Move your hand to the right." }。
前端接收 -> 更新图片 -> 播放 "Move your hand to the right."。

5. 开发与部署要求

目录结构: 清晰的前后端分离目录 (例如 /backend 和 /frontend)。
依赖管理: 后端使用 requirements.txt，前端使用 package.json。
跨域处理: 后端需配置 CORS 允许前端访问。