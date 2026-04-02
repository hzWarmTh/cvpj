import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';
import useVoiceInteraction from './useVoiceInteraction';

let _msgId = 0;
const nextId = () => ++_msgId;

export default function App() {
  const wsRef = useRef(null);
  const lastCommandRef = useRef('');
  const chatEndRef = useRef(null);
  // 转发 hook 暴露的 TTS 暂停/恢复函数，避免 speakOnce 与 hook 之间的循环依赖
  const pauseForTTSRef = useRef(null);
  const resumeFromTTSRef = useRef(null);

  const [displayImage, setDisplayImage] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | connecting | running | error
  const [message, setMessage] = useState('');
  const [instruction, setInstruction] = useState('');
  const [targetObject, setTargetObject] = useState('');
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [rotation, setRotation] = useState(0);
  const [chatHistory, setChatHistory] = useState([]); // [{id, role:'user'|'tom', text}]
  const [grabState, setGrabState] = useState('searching');

  // auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // ---- TTS (read each instruction twice) ----
  const speak = useCallback((text) => {
    if (!text || text === lastCommandRef.current) return;
    lastCommandRef.current = text;
    window.speechSynthesis.cancel();
    const utt1 = new SpeechSynthesisUtterance(text);
    utt1.lang = 'en-US';
    utt1.rate = 1.1;
    const utt2 = new SpeechSynthesisUtterance(text);
    utt2.lang = 'en-US';
    utt2.rate = 1.1;
    window.speechSynthesis.speak(utt1);
    window.speechSynthesis.speak(utt2);
  }, []);

  // ---- TTS for voice responses (single play) + append to chat ----
  // 通过 ref 调用 hook 的 pause/resume，避免循环依赖
  const speakOnce = useCallback((text) => {
    if (!text) return;
    window.speechSynthesis.cancel();
    // 暂停识别，防止 TTS 输出被麦克风录入
    pauseForTTSRef.current?.();
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = 'en-US';
    utt.rate = 1.0;
    utt.onend = () => resumeFromTTSRef.current?.();
    // 若 TTS 因某种原因中断，也要恢复识别
    utt.onerror = () => resumeFromTTSRef.current?.();
    window.speechSynthesis.speak(utt);
    setChatHistory(prev => [...prev, { id: nextId(), role: 'tom', text }]);
  }, []);

  // ---- select target via WebSocket ----
  const selectTarget = useCallback((name) => {
    console.log('selectTarget called with:', name);
    setTargetObject(name);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ target: name }));
      console.log('Sent target to video WebSocket:', name);
    } else {
      console.log('Video WebSocket not connected');
    }
  }, []);

  // ---- Voice Interaction Hook ----
  const {
    isListening,
    isAwake,
    isSpeaking,
    isProcessing,
    lastTranscription,
    voiceStatus,
    error: voiceError,
    startListening,
    stopListening,
    pauseForTTS,
    resumeFromTTS,
  } = useVoiceInteraction({
    onTargetSelected: selectTarget,
    onResponse: speakOnce,
  });

  // 将 hook 函数同步到 ref，供 speakOnce 使用（每次渲染更新）
  pauseForTTSRef.current = pauseForTTS;
  resumeFromTTSRef.current = resumeFromTTS;

  // ---- cycle rotation ----
  const cycleRotation = useCallback(() => {
    setRotation((prev) => {
      const next = (prev + 90) % 360;
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ rotation: next }));
      }
      return next;
    });
  }, []);

  // ---- start detection ----
  const handleStart = useCallback(() => {
    setStatus('connecting');
    setMessage('connecting to backend (DroidCam)...');

    const ws = new WebSocket('ws://localhost:8000/ws/video');
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('running');
      setMessage('connected, detecting in real-time');
      ws.send(JSON.stringify({ rotation }));
      if (targetObject) {
        ws.send(JSON.stringify({ target: targetObject }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setDisplayImage(data.image);
        setInstruction(data.display_instruction || data.instruction || '');
        setGrabState(data.grab_state || 'searching');

        if (data.detected_objects) {
          setDetectedObjects(data.detected_objects);
          if (data.target) {
            setTargetObject(data.target);
          }
        }

        // 暂时禁用指令语音播报
        const speechInstruction = data.speech_instruction || '';
        if (speechInstruction && !isListening) {
          speak(speechInstruction);
        }
      } catch {
        setDisplayImage(event.data);
      }
    };

    ws.onerror = () => {
      setMessage('WebSocket error - is the backend running?');
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === 'error' ? 'error' : 'idle'));
    };
  }, [speak, rotation, targetObject]);

  // ---- stop detection ----
  const handleStop = useCallback(() => {
    lastCommandRef.current = '';
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setDisplayImage(null);
    setStatus('idle');
    setMessage('stopped');
    setInstruction('');
    setDetectedObjects([]);
    setGrabState('searching');
    window.speechSynthesis.cancel();
  }, []);

  // append user utterances to chat history
  useEffect(() => {
    if (lastTranscription) {
      setChatHistory(prev => [...prev, { id: nextId(), role: 'user', text: lastTranscription }]);
    }
  }, [lastTranscription]);

  // ---- cleanup on unmount ----
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const isRunning = status === 'running' || status === 'connecting';

  // Voice status display helper
  const getVoiceStatusText = () => {
    if (!isListening) return 'Voice Off';
    if (isProcessing) return 'Processing...';
    if (isSpeaking) return 'Listening...';
    if (isAwake) return 'Awake - Say a command';
    return 'Say "Hi Tom"';
  };

  const getVoiceStatusClass = () => {
    if (!isListening) return '';
    if (isProcessing) return 'processing';
    if (isSpeaking) return 'speaking';
    if (isAwake) return 'awake';
    return 'listening';
  };

  return (
    <div className="app-container">
      {/* Left: video stream */}
      <div className="video-panel">
        <div className="video-header">
          <h2 className="panel-title">Real-time Video</h2>
          <button className="btn-rotate" onClick={cycleRotation} title="Rotate">
            {" " + rotation + ""}
          </button>
        </div>
        <div className="video-wrapper">
          {displayImage ? (
            <img src={displayImage} alt="processed frame" className="video-img" />
          ) : (
            <div className="video-placeholder">
              {status === 'connecting'
                ? 'Connecting...'
                : 'Click "Start" to begin'}
            </div>
          )}
        </div>
        {instruction && (
          <div className={`video-command-bar${grabState === 'grabbed' ? ' grabbed' : grabState === 'close' ? ' close' : ''}`}>
            <span className={`video-command-text${grabState === 'grabbed' ? ' grabbed' : grabState === 'close' ? ' close' : ''}`}
                  style={{ fontSize: '1.6rem', fontWeight: 'bold' }}>
              {instruction}
            </span>
          </div>
        )}
      </div>

      {/* Right: control panel */}
      <div className="control-panel">
        <div className="control-header">
          <h1 className="app-title">Visual Assist</h1>
          <div className="status-pill">
            <span className={"status-dot" + (isRunning ? " active" : "")} />
            <span>{isRunning ? 'Running' : 'Stopped'}</span>
          </div>
        </div>

        {/* Start / Stop */}
        <div className="btn-group">
          <button className="btn btn-start" disabled={isRunning} onClick={handleStart}>
            {status === 'connecting' ? 'Connecting...' : 'Start'}
          </button>
          <button className="btn btn-stop" disabled={!isRunning} onClick={handleStop}>
            Stop
          </button>
        </div>

        {/* Target selection */}
        <div className="target-card">
          <div className="target-card-header">
            <h3>Target Object</h3>
            {targetObject && (
              <span className="target-badge">{targetObject}</span>
            )}
          </div>
          {detectedObjects.length > 0 ? (
            <div className="object-chips">
              {detectedObjects.map((name) => (
                <button
                  key={name}
                  className={"chip" + (name === targetObject ? " chip-active" : "")}
                  onClick={() => selectTarget(name)}
                >
                  {name}
                </button>
              ))}
            </div>
          ) : (
            <p className="target-hint">
              {isRunning ? 'Waiting for detections...' : 'Detected objects will appear here after starting'}
            </p>
          )}
        </div>

        {/* Voice Control */}
        <div className="voice-card">
          <div className="voice-card-top">
            <div className={`voice-status ${getVoiceStatusClass()}`}>
              <span className="voice-indicator"></span>
              <span>{getVoiceStatusText()}</span>
            </div>
            <button
              className={`btn-voice-toggle ${isListening ? 'active' : ''}`}
              onClick={isListening ? stopListening : startListening}
            >
              {isListening ? '■ Stop' : '🎤 Start'}
            </button>
          </div>
          <p className="voice-hint">
            {isAwake
              ? '"Where are the [object]?" · "Did I get it?"'
              : 'Say "Hi Tom" to wake up'}
          </p>
          {voiceError && <p className="voice-error">{voiceError}</p>}
        </div>

        {/* Conversation History */}
        <div className="chat-card">
          <h3 className="chat-title">Conversation</h3>
          <div className="chat-log">
            {chatHistory.length === 0 ? (
              <p className="chat-empty">No conversation yet.</p>
            ) : (
              chatHistory.map((msg) => (
                <div key={msg.id} className={`chat-row chat-row-${msg.role}`}>
                  <div className={`chat-bubble chat-bubble-${msg.role}`}>
                    <span className="chat-role">{msg.role === 'user' ? 'You' : 'Tom'}</span>
                    <span className="chat-text">{msg.text}</span>
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
