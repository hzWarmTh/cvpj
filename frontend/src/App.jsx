import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

export default function App() {
  const wsRef = useRef(null);
  const lastCommandRef = useRef(''); // 上一次播报的指令（用于去重）

  const [displayImage, setDisplayImage] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | connecting | running | error
  const [message, setMessage] = useState('');
  const [instruction, setInstruction] = useState('');

  // ---- TTS 语音播报（仅在指令变化时触发，每条读两遍） ----
  const speak = useCallback((text) => {
    if (!text || text === lastCommandRef.current) return;
    lastCommandRef.current = text;
    window.speechSynthesis.cancel();
    // 第一遍
    const utt1 = new SpeechSynthesisUtterance(text);
    utt1.lang = 'en-US';
    utt1.rate = 1.1;
    // 第二遍
    const utt2 = new SpeechSynthesisUtterance(text);
    utt2.lang = 'en-US';
    utt2.rate = 1.1;
    window.speechSynthesis.speak(utt1);
    window.speechSynthesis.speak(utt2);
  }, []);

  // ---- 开始 ----
  const handleStart = useCallback(() => {
    setStatus('connecting');
    setMessage('正在连接后端（DroidCam 摄像头）…');

    const ws = new WebSocket('ws://localhost:8000/ws/video');
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('running');
      setMessage('已连接 DroidCam，实时检测中');
    };

    ws.onmessage = (event) => {
      // 后端返回 JSON: {image, instruction}
      try {
        const data = JSON.parse(event.data);
        setDisplayImage(data.image);
        setInstruction(data.instruction || '');
        // 仅当指令变化时语音播报（每条读两遍）
        speak(data.instruction);
      } catch {
        // 兼容纯 Base64 回退
        setDisplayImage(event.data);
      }
    };

    ws.onerror = () => {
      setMessage('WebSocket 连接出错，请确认后端已启动');
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === 'error' ? 'error' : 'idle'));
    };
  }, [speak]);

  // ---- 停止 ----
  const handleStop = useCallback(() => {
    lastCommandRef.current = '';
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setDisplayImage(null);
    setStatus('idle');
    setMessage('已停止');
    setInstruction('');
    window.speechSynthesis.cancel();
  }, []);

  // ---- 组件卸载时清理 ----
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // ---- UI ----
  const isRunning = status === 'running' || status === 'connecting';

  return (
    <div className="app-container">
      {/* 左侧：视频流 */}
      <div className="video-panel">
        <h2 className="panel-title">实时视频流</h2>
        <div className="video-wrapper">
          {displayImage ? (
            <img src={displayImage} alt="处理后的画面" className="video-img" />
          ) : (
            <div className="video-placeholder">
              {status === 'connecting'
                ? '正在连接…'
                : '点击右侧「开始检测」启动'}
            </div>
          )}
        </div>
        {/* 画面下方指令条 */}
        {instruction && (
          <div className="video-command-bar">
            <span className="video-command-text" style={{ fontSize: '1.6rem', fontWeight: 'bold' }}>
              {instruction}
            </span>
          </div>
        )}
      </div>

      {/* 右侧：控制面板 */}
      <div className="control-panel">
        <h1 className="app-title">视觉辅助系统</h1>

        <div className="status-bar">
          <span className={`status-dot ${isRunning ? 'active' : ''}`} />
          <span>{isRunning ? '运行中' : '已停止'}</span>
        </div>

        {message && <p className="message">{message}</p>}

        {instruction && (
          <div className="command-card">
            <span className="command-label">指令</span>
            <span className="command-text" style={{ fontSize: '1.4rem', fontWeight: 'bold' }}>
              {instruction}
            </span>
          </div>
        )}

        <div className="btn-group">
          <button
            className="btn btn-start"
            disabled={isRunning}
            onClick={handleStart}
          >
            {status === 'connecting' ? '连接中…' : '开始检测'}
          </button>
          <button
            className="btn btn-stop"
            disabled={!isRunning}
            onClick={handleStop}
          >
            停止检测
          </button>
        </div>

        <div className="info-card">
          <h3>使用说明</h3>
          <ol>
            <li>确保后端已运行在 <code>localhost:8000</code></li>
            <li>确保手机 DroidCam 已开启</li>
            <li>点击「开始检测」连接后端</li>
            <li>画面将实时显示 YOLOv8 物体检测 &amp; 手部骨骼</li>
            <li>点击「停止检测」结束</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
