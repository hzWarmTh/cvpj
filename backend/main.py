"""视障人士视觉辅助系统 - 后端入口"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 配置日志（必须在其他模块导入前执行）
logging.basicConfig(level=logging.INFO)

# 导入模块（触发模型加载）
import models  # noqa: E402
from routes import router  # noqa: E402

# FastAPI 应用初始化
app = FastAPI(title="视障人士视觉辅助系统 - 后端")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    models.hands.close()
    logging.getLogger(__name__).info("应用已关闭，资源已释放")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
