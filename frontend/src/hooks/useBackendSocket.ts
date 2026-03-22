import { useEffect, useMemo, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

import { BACKEND_WS_URL } from '../config/backend';
import type { BackendState } from '../types/backendState';
import type { BackendSocketMessage } from '../types/wsProtocol';
import { useWsProtocol } from './useWsProtocol';

export function useBackendSocket(backendReady: boolean) {
  const [currentState, setCurrentState] = useState<BackendState | null>(null);

  const { sendMessage, lastJsonMessage: rawMessage, readyState } = useWebSocket<BackendSocketMessage>(
    BACKEND_WS_URL,
    {
      shouldReconnect: () => true,
      reconnectInterval: 2000,
      reconnectAttempts: Infinity,
      heartbeat: {
        message: JSON.stringify({ type: 'ping' }),
        interval: 10000,
        timeout: 30000,
      },
      share: false,
    },
    backendReady,
  );

  const parsed = useWsProtocol(rawMessage, currentState);

  useEffect(() => {
    if (parsed.state) {
      setCurrentState(parsed.state);
    }
  }, [parsed.state]);

  const sendReset = () => {
    sendMessage(JSON.stringify({ type: 'reset' }));
  };

  return useMemo(() => ({
    socketUrl: BACKEND_WS_URL,
    readyState,
    isConnected: readyState === ReadyState.OPEN,
    lastJsonMessage: currentState,
    protocolVersion: parsed.protocolVersion,
    lastEvent: parsed.lastEvent,
    sendReset,
  }), [currentState, parsed.lastEvent, parsed.protocolVersion, readyState]);
}
