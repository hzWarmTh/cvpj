/**
 * 语音交互 Hook
 * 使用浏览器原生 Web Speech API 进行语音识别
 */

import { useState, useRef, useCallback, useEffect } from 'react';

const VOICE_WS_URL = 'ws://localhost:8000/ws/voice';

export function useVoiceInteraction({ onTargetSelected, onResponse }) {
  const [isListening, setIsListening] = useState(false);
  const [isAwake, setIsAwake] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastTranscription, setLastTranscription] = useState('');
  const [voiceStatus, setVoiceStatus] = useState('idle');
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const recognitionRef = useRef(null);
  const isListeningRef = useRef(false);  // 使用 ref 避免闪包问题
  const isAwakeRef = useRef(false);

  // 同步 ref
  useEffect(() => {
    isListeningRef.current = isListening;
  }, [isListening]);
  
  useEffect(() => {
    isAwakeRef.current = isAwake;
  }, [isAwake]);

  // 唤醒词列表
  const WAKE_WORDS = ['hey tom', 'hi tom', 'hello tom', 'tom', 'wake up', 'hello', 'hey'];
  
  // 停止词
  const STOP_WORDS = ['stop', 'cancel', 'quit', 'exit', 'sleep'];

  // 检查是否匹配唤醒词
  const checkWakeWord = useCallback((text) => {
    const lower = text.toLowerCase().trim();
    return WAKE_WORDS.some(word => lower.includes(word));
  }, []);

  // 检查停止词
  const checkStopWord = useCallback((text) => {
    const lower = text.toLowerCase().trim();
    return STOP_WORDS.some(word => lower.includes(word));
  }, []);

  // 处理语音识别结果
  const handleSpeechResult = useCallback((text) => {
    if (!text) return;
    
    setLastTranscription(text);
    console.log('Speech recognized:', text);

    // 检查唤醒词
    if (checkWakeWord(text)) {
      setIsAwake(true);
      if (onResponse) {
        onResponse("I'm listening");
      }
      // 发送到后端同步唤醒状态
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'wake' }));
      }
      return;
    }

    // 如果未唤醒，忽略
    if (!isAwake) {
      return;
    }

    // 检查停止词
    if (checkStopWord(text)) {
      setIsAwake(false);
      if (onResponse) {
        onResponse("Okay.");
      }
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'sleep' }));
      }
      return;
    }

    // 发送到后端处理意图，回答后自动休眠
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        type: 'text',
        text: text 
      }));
    }
  }, [isAwake, checkWakeWord, checkStopWord, onResponse]);

  // 初始化 Web Speech API
  const initSpeechRecognition = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser');
      return null;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const last = event.results.length - 1;
      const text = event.results[last][0].transcript;
      setIsSpeaking(false);
      setIsProcessing(true);
      handleSpeechResult(text);
      setIsProcessing(false);
    };

    recognition.onspeechstart = () => {
      setIsSpeaking(true);
      setVoiceStatus('speaking');
    };

    recognition.onspeechend = () => {
      setIsSpeaking(false);
      setVoiceStatus(isAwakeRef.current ? 'awake' : 'listening');
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        setError(`Speech error: ${event.error}`);
      }
      // 出错后也尝试重启
      if (event.error === 'no-speech' && isListeningRef.current) {
        setTimeout(() => {
          if (recognitionRef.current && isListeningRef.current) {
            try {
              recognitionRef.current.start();
              console.log('Speech recognition restarted after no-speech');
            } catch (e) {
              // 忽略
            }
          }
        }, 100);
      }
    };

    recognition.onend = () => {
      console.log('Speech recognition ended, isListening:', isListeningRef.current);
      // 自动重启（使用 ref 避免闭包问题）
      if (isListeningRef.current && recognitionRef.current) {
        setTimeout(() => {
          if (recognitionRef.current && isListeningRef.current) {
            try {
              recognitionRef.current.start();
              console.log('Speech recognition restarted');
            } catch (e) {
              console.error('Failed to restart speech recognition:', e);
            }
          }
        }, 100);
      }
    };

    return recognition;
  }, [handleSpeechResult]);  // 移除 isAwake, isListening 依赖，使用 ref 代替

  // 连接 WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(VOICE_WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Voice WebSocket connected');
        setError(null);
        resolve();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'intent') {
            // 播放响应
            if (data.response && onResponse) {
              onResponse(data.response);
            }
            // 如果选择了目标
            if (data.action === 'select_target' && data.target && onTargetSelected) {
              onTargetSelected(data.target);
            }
            // 更新唤醒状态
            if (data.awake !== undefined) {
              setIsAwake(data.awake);
            }
          }
        } catch (e) {
          console.error('Failed to parse voice message:', e);
        }
      };

      ws.onerror = (e) => {
        console.error('Voice WebSocket error:', e);
        reject(e);
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    });
  }, [onTargetSelected, onResponse]);

  // 开始语音监听
  const startListening = useCallback(async () => {
    setError(null);
    
    try {
      // 连接 WebSocket
      await connectWebSocket();
      
      // 初始化语音识别
      const recognition = initSpeechRecognition();
      if (!recognition) return;
      
      recognitionRef.current = recognition;
      recognition.start();
      
      setIsListening(true);
      setVoiceStatus('listening');
    } catch (e) {
      console.error('Failed to start listening:', e);
      setError('Failed to start voice input');
    }
  }, [connectWebSocket, initSpeechRecognition]);

  // 停止语音监听
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsListening(false);
    setIsSpeaking(false);
    setIsProcessing(false);
    setVoiceStatus('idle');
    setIsAwake(false);
  }, []);

  // 清理
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  // 更新 recognition 的回调（当 isAwake 或 isListening 变化时）
  useEffect(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onresult = (event) => {
        const last = event.results.length - 1;
        const text = event.results[last][0].transcript;
        setIsSpeaking(false);
        handleSpeechResult(text);
      };
    }
  }, [handleSpeechResult]);

  return {
    isListening,
    isAwake,
    isSpeaking,
    isProcessing,
    lastTranscription,
    voiceStatus,
    error,
    startListening,
    stopListening,
  };
}

export default useVoiceInteraction;
