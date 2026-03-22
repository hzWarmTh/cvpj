import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

export default function App() {
  const wsRef = useRef(null);
  const lastCommandRef = useRef('');

  const [displayImage, setDisplayImage] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | connecting | running | error
  const [message, setMessage] = useState('');
  const [instruction, setInstruction] = useState('');
  const [targetObject, setTargetObject] = useState('');
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [rotation, setRotation] = useState(0);
  const [grabState, setGrabState] = useState('searching');

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

  // ---- select target via WebSocket ----
  const selectTarget = useCallback((name) => {
    setTargetObject(name);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ target: name }));
    }
  }, []);

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

        const speechInstruction = data.speech_instruction || '';
        if (speechInstruction) {
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

  // ---- cleanup on unmount ----
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const isRunning = status === 'running' || status === 'connecting';

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
        <h1 className="app-title">Visual Assist</h1>

        {/* Target selection */}
        <div className="target-card">
          <h3>Target Object</h3>
          {detectedObjects.length > 0 ? (
            <>
              <p className="target-hint">Select an object to track:</p>
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
            </>
          ) : (
            <p className="target-hint">
              {isRunning ? 'Waiting for detections...' : 'Detected objects will appear here after starting'}
            </p>
          )}
          {targetObject && (
            <p className="target-tip">
              Current target: <strong>{targetObject}</strong>
            </p>
          )}
        </div>

        <div className="status-bar">
          <span className={"status-dot" + (isRunning ? " active" : "")} />
          <span>{isRunning ? 'Running' : 'Stopped'}</span>
        </div>

        {message && <p className="message">{message}</p>}

        {instruction && (
          <div className="command-card">
            <span className="command-label">Command</span>
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
            {status === 'connecting' ? 'Connecting...' : 'Start'}
          </button>
          <button
            className="btn btn-stop"
            disabled={!isRunning}
            onClick={handleStop}
          >
            Stop
          </button>
        </div>

        <div className="info-card">
          <h3>Instructions</h3>
          <ol>
            <li>Backend must be running on <code>localhost:8000</code></li>
            <li>DroidCam must be active on your phone</li>
            <li>Click Start to connect</li>
            <li>Detected objects appear as selectable buttons</li>
            <li>Click an object name to set it as the target</li>
            <li>Use the rotate button if the image is flipped</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
