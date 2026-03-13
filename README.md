# 视障人士视觉辅助物体抓取系统

一个 B/S 架构的视觉辅助系统，帮助视障用户通过实时语音引导准确抓取物体。

## 技术栈

### 后端
- **语言**: Python 3.10+
- **框架**: FastAPI + Uvicorn
- **计算机视觉**: 
  - OpenCV: 图像编解码
  - MediaPipe: 手部关键点检测
  - YOLOv8: 物体检测
- **通信**: WebSocket 实时通信

### 前端
- **框架**: React 18
- **构建工具**: Vite
- **样式**: Tailwind CSS
- **核心库**:
  - react-webcam: 摄像头采集
  - WebSocket: 实时数据传输
  - Browser TTS: 语音合成

## 项目结构

```
.
├── backend/                    # 后端目录
│   ├── main.py                # FastAPI 入口应用
│   └── requirements.txt        # Python 依赖
├── frontend/                   # 前端目录
│   ├── src/
│   │   ├── App.jsx            # 主应用组件
│   │   ├── App.css            # 应用样式
│   │   ├── main.jsx           # React 入口
│   │   └── index.css          # 全局样式
│   ├── public/
│   │   └── index.html         # HTML 模板
│   ├── package.json           # npm 依赖
│   ├── vite.config.js         # Vite 配置
│   ├── tailwind.config.js     # Tailwind 配置
│   ├── postcss.config.js      # PostCSS 配置
│   └── .gitignore            # Git 忽略文件
└── README.md                  # 本文件
```

## 快速启动

### 1. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

> **注意**: 首次运行时，YOLOv8 模型会自动下载 (~100MB)

### 2. 启动后端服务

```bash
cd backend
python main.py
```

后端将在 `http://localhost:8000` 启动。你可以访问 `http://localhost:8000/docs` 查看 API 文档。

### 3. 安装前端依赖

在新的终端窗口：

```bash
cd frontend
npm install
```

### 4. 启动前端开发服务器

```bash
cd frontend
npm run dev
```

前端将在 `http://localhost:3000` 启动。

## 使用流程

1. **访问前端**: 打开浏览器访问 `http://localhost:3000`
2. **允许权限**: 允许网页访问摄像头和麦克风
3. **设置目标**: 在控制面板输入要抓取的物体名称（如 "cup", "bottle" 等）
4. **开始引导**: 点击"开始引导"按钮
5. **按照指导**: 根据语音提示移动手部，靠近目标物体
6. **执行抓取**: 听到"现在可以抓取了"提示后进行抓取

## API 接口

### 设置目标物体

```
POST /set-target
Content-Type: application/json

{
  "target": "cup"
}
```

### WebSocket 通信

```
ws://localhost:8000/ws/guidance
```

**发送数据**:
```json
{
  "frame": "data:image/jpeg;base64,..."
}
```

**接收数据**:
```json
{
  "image": "data:image/jpeg;base64,...",
  "command": "move_left",
  "message": "请向左移动你的手。",
  "target": "cup",
  "frame_info": {
    "target": {
      "center_x": 320,
      "center_y": 240,
      "confidence": 0.95
    },
    "hand": {
      "center_x": 350,
      "center_y": 200,
      "detected": true
    }
  }
}
```

## 生产环境构建

### 前端构建

```bash
cd frontend
npm run build
```

构建后的文件将在 `frontend/dist` 目录中。

### 后端部署

使用 Gunicorn 部署：

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

## 常见问题

### WebSocket 连接失败
- 检查后端是否正常运行
- 确保防火墙允许 8000 端口
- 查看浏览器控制台是否有错误信息

### 摄像头权限被拒绝
- 检查浏览器权限设置
- 在 macOS 上，需要在"系统偏好设置 > 安全性与隐私"中授予权限

### YOLOv8 模型下载缓慢
- 可以手动下载模型: `yolo detect predict model=yolov8n.pt source=0`
- 或修改 main.py 中的模型路径为本地路径

### 语音识别不工作
- 确保麦克风被允许
- 在某些浏览器（如 Safari）中可能需要额外配置

## 依赖版本

后端依赖版本详见 `backend/requirements.txt`

前端依赖版本详见 `frontend/package.json`

## License

本项目用于教学目的。

## 支持

如有问题，请检查：
1. 各依赖是否正确安装
2. 前后端服务是否启动
3. 浏览器控制台的错误信息
4. 后端服务的日志输出
