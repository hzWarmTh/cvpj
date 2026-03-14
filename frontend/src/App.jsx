import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

export default function App() {
  const wsRef = useRef(null);
  const lastCommandRef = useRef(''); // 上一次播报的指令（用于去重）

  const [displayImage, setDisplayImage] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | connecting | running | error
  const [message, setMessage] = useState('');
  const [instruction, setInstruction] = useState('');
  const [targetInput, setTargetInput] = useState('cell phone');
  const [targetObject, setTargetObject] = useState('cell phone');
  const [isSettingTarget, setIsSettingTarget] = useState(false);
  const [targetDirty, setTargetDirty] = useState(false);

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

  // ---- 提交目标物体（检测前设置） ----
  const handleApplyTarget = useCallback(async () => {
    const nextTarget = targetInput.trim();
    if (!nextTarget) {
      setMessage('目标物体不能为空');
      return;
    }

    setIsSettingTarget(true);
    setMessage('正在设置目标物体...');

    try {
      const res = await fetch('http://localhost:8000/set-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: nextTarget }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      const appliedTarget = (data.target || nextTarget).trim();
      setTargetObject(appliedTarget);
      setTargetInput(appliedTarget);
      setTargetDirty(false);
      setMessage(`目标已设置为: ${appliedTarget}`);
    } catch (err) {
      setMessage(`设置目标失败，请确认后端可用 (${err.message})`);
    } finally {
      setIsSettingTarget(false);
    }
  }, [targetInput]);

  // ---- 开始 ----
  const handleStart = useCallback(() => {
    if (targetDirty) {
      setMessage('请先点击“应用目标”，再开始检测');
      return;
    }

    setStatus('connecting');
    setMessage('正在连接后端（DroidCam 摄像头）…');

    const ws = new WebSocket('ws://localhost:8000/ws/video');
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('running');
      setMessage('已连接 DroidCam，实时检测中');
    };

    ws.onmessage = (event) => {
      // 后端返回 JSON: {image, display_instruction, speech_instruction}
      try {
        const data = JSON.parse(event.data);
        setDisplayImage(data.image);
        setInstruction(data.display_instruction || data.instruction || '');

        // 语音播报仅使用冷却后的 speech_instruction
        const speechInstruction = data.speech_instruction || '';
        if (speechInstruction) {
          speak(speechInstruction);
        }
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
  }, [speak, targetDirty]);

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
  const canEditTarget = !isRunning && !isSettingTarget;

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

        <div className="target-card">
          <h3>目标物体</h3>
          <div className="target-row">
            <input
              className="target-input"
              type="text"
              value={targetInput}
              onChange={(e) => {
                setTargetInput(e.target.value);
                setTargetDirty(true);
              }}
              placeholder="例如: cup / bottle / cell phone"
              disabled={!canEditTarget}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && canEditTarget) {
                  handleApplyTarget();
                }
              }}
            />
            <button
              className="btn btn-apply"
              onClick={handleApplyTarget}
              disabled={!canEditTarget}
            >
              {isSettingTarget ? '设置中…' : '应用目标'}
            </button>
          </div>
          <p className="target-tip">
            当前目标: <strong>{targetObject}</strong>
          </p>
          {targetDirty && !isRunning && (
            <p className="target-warning">你修改了目标，请先应用后再开始检测。</p>
          )}
        </div>

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
