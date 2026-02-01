import React, { createContext, useContext, useState, type ReactNode } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

// --- Types ---

export type GameState = 'ARMED' | 'TRACKING' | 'VIRTUAL_ROLLING' | 'STOPPED' | 'COOLDOWN';

export interface BallData {
  x_px: number;
  y_px: number;
  radius_px: number;
  confidence: number;
}

export interface VelocityData {
  vx_px_s: number;
  vy_px_s: number;
  speed_px_s: number;
}

export interface ShotResult {
  speed_m_s: number;
  speed_px_s: number;
  direction_deg: number;
  physical_distance_m: number;
  virtual_distance_m: number;
  distance_m: number;
  distance_px: number;
  trajectory: number[][]; // [x, y]
  exited_frame: boolean;
}

export interface PredictionData {
  trajectory: number[][];
  final_position: [number, number];
  final_time_s: number;
  exit_speed_px_s: number;
}

export interface Metrics {
  cap_fps: number;
  proc_fps: number;
  disp_fps: number;
  proc_latency_ms: number;
  idle_stddev: number;
}

export interface VirtualBall {
  x: number;
  y: number;
  vx: number;
  vy: number;
  speed_m_s: number;
  distance_m: number;
  is_rolling: boolean;
}

export interface BackendState {
  timestamp_ms: number;
  state: GameState;
  lane: string;
  ball: BallData | null;
  ball_visible: boolean;
  velocity: VelocityData | null;
  prediction: PredictionData | null;
  virtual_ball: VirtualBall | null;
  shot: ShotResult | null;
  metrics: Metrics;
  calibrated: boolean;
  auto_calibrated: boolean;
  lens_calibrated: boolean;
  pixels_per_meter: number;
  overlay_radius_scale: number;
  resolution: [number, number];
}

interface WebSocketContextType {
  readyState: ReadyState;
  lastJsonMessage: BackendState | null;
  isConnected: boolean;
  // Convenience accessors
  gameState: GameState;
  ballPosition: { x: number; y: number } | null;
  pixelsPerMeter: number;
  sendReset: () => void;
}

// --- Context ---

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// --- Provider ---

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socketUrl] = useState('ws://localhost:8000/ws');
  
  const {
    sendMessage,
    lastJsonMessage,
    readyState,
  } = useWebSocket<BackendState>(socketUrl, {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
    reconnectAttempts: 20,
    share: false, // share: true can cause loops in strict mode or single provider setups
  });

  const isConnected = readyState === ReadyState.OPEN;

  // Convenience helpers
  const gameState = lastJsonMessage?.state || 'ARMED';
  const ballPosition = lastJsonMessage?.ball 
    ? { x: lastJsonMessage.ball.x_px, y: lastJsonMessage.ball.y_px } 
    : (lastJsonMessage?.virtual_ball ? { x: lastJsonMessage.virtual_ball.x, y: lastJsonMessage.virtual_ball.y } : null);
  const pixelsPerMeter = lastJsonMessage?.pixels_per_meter || 1150;

  const sendReset = () => {
    sendMessage(JSON.stringify({ type: 'reset' }));
  };

  const value: WebSocketContextType = {
    readyState,
    lastJsonMessage,
    isConnected,
    gameState,
    ballPosition,
    pixelsPerMeter,
    sendReset,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

// --- Hook ---

export const usePuttingState = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('usePuttingState must be used within a WebSocketProvider');
  }
  return context;
};
