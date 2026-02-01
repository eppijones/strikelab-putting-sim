import React, { createContext, useContext, useState, useEffect, useRef, useCallback, type ReactNode } from 'react';
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

export type ShotResultType = 'pending' | 'made' | 'miss_short' | 'miss_long' | 'miss_left' | 'miss_right' | 'lip_out';

export interface GameState_Data {
  hole: {
    distance_m: number;
    position_x_m: number;
    position_y_m: number;
    radius_m: number;
  };
  last_shot: {
    result: ShotResultType;
    distance_to_hole_m: number;
    lateral_miss_m: number;
    depth_miss_m: number;
    miss_description: string;
    is_made: boolean;
  } | null;
}

export interface SessionData {
  session_id: string;
  duration_s: number;
  total_putts: number;
  putts_made: number;
  make_percentage: number;
  current_streak: number;
  best_streak: number;
  avg_speed_m_s: number;
  avg_miss_distance_m: number;
  putts_by_distance: Record<string, { total: number; made: number; percentage: number }>;
}

export type DrillType = 'none' | 'distance_control' | 'ladder_drill';

export interface DrillData {
  active: boolean;
  drill_type: DrillType;
  current_target_m?: number;
  total_points?: number;
  attempts?: number;
  targets_completed?: number;
  duration_s?: number;
  ladder_position?: number;
  last_attempt?: {
    rating: string;
    points: number;
    error_cm: number;
  };
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
  // Game logic and session data
  game?: GameState_Data;
  session?: SessionData;
  drill?: DrillData;
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
  // Test shot
  triggerTestShot: () => void;
  isTestShotActive: boolean;
  // Game and session data
  gameData: GameState_Data | null;
  sessionData: SessionData | null;
  drillData: DrillData | null;
  setHoleDistance: (distance_m: number) => Promise<void>;
  resetSession: () => Promise<void>;
  startDrill: (drillType: DrillType) => Promise<void>;
  stopDrill: () => Promise<void>;
}

// --- Test Shot Generator ---

interface TestShotState {
  active: boolean;
  startTime: number;
  phase: GameState;
  shot: ShotResult;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  currentX: number;
  currentY: number;
  duration: number; // total animation duration in ms
}

function generateTestShot(pixelsPerMeter: number): TestShotState {
  // Generate realistic random shot parameters
  const speed_m_s = 1.5 + Math.random() * 2.5; // 1.5 - 4.0 m/s
  const direction_deg = (Math.random() - 0.5) * 10; // -5 to +5 degrees
  const distance_m = 2 + Math.random() * 6; // 2 - 8 meters
  
  // Convert to pixels
  const speed_px_s = speed_m_s * pixelsPerMeter;
  const distance_px = distance_m * pixelsPerMeter;
  
  // Starting position (near camera, center of frame)
  const startX = 100; // pixels from camera
  const startY = 400; // center of 800px height
  
  // End position based on direction and distance
  const direction_rad = (direction_deg * Math.PI) / 180;
  const endX = startX + distance_px * Math.cos(direction_rad);
  const endY = startY + distance_px * Math.sin(direction_rad);
  
  // Generate trajectory points
  const numPoints = 50;
  const trajectory: number[][] = [];
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    const x = startX + (endX - startX) * t;
    const y = startY + (endY - startY) * t;
    trajectory.push([x, y]);
  }
  
  // Animation duration based on distance and speed (with deceleration)
  const duration = (distance_m / speed_m_s) * 2000; // roughly 2x the simple time for deceleration
  
  return {
    active: true,
    startTime: Date.now(),
    phase: 'TRACKING',
    shot: {
      speed_m_s,
      speed_px_s,
      direction_deg,
      physical_distance_m: distance_m * 0.3, // physical portion
      virtual_distance_m: distance_m * 0.7, // virtual portion  
      distance_m,
      distance_px,
      trajectory,
      exited_frame: true,
    },
    startX,
    startY,
    endX,
    endY,
    currentX: startX,
    currentY: startY,
    duration,
  };
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
    lastJsonMessage: realLastJsonMessage,
    readyState,
  } = useWebSocket<BackendState>(socketUrl, {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
    reconnectAttempts: 20,
    share: false, // share: true can cause loops in strict mode or single provider setups
  });

  const isConnected = readyState === ReadyState.OPEN;
  const realPixelsPerMeter = realLastJsonMessage?.pixels_per_meter || 1150;

  // --- Test Shot State ---
  const [testShot, setTestShot] = useState<TestShotState | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const triggerTestShot = useCallback(() => {
    // Cancel any existing test shot
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    const newTestShot = generateTestShot(realPixelsPerMeter);
    setTestShot(newTestShot);
    console.log('[TestShot] Triggered:', {
      speed: newTestShot.shot.speed_m_s.toFixed(2) + ' m/s',
      direction: newTestShot.shot.direction_deg.toFixed(1) + '°',
      distance: newTestShot.shot.distance_m.toFixed(2) + ' m',
    });
  }, [realPixelsPerMeter]);

  // Test shot animation loop
  useEffect(() => {
    if (!testShot?.active) return;

    const animate = () => {
      const elapsed = Date.now() - testShot.startTime;
      const trackingDuration = testShot.duration * 0.3; // First 30% is tracking
      const virtualDuration = testShot.duration * 0.7; // Next 70% is virtual rolling
      const stopDuration = 2000; // Show stopped for 2s
      const cooldownDuration = 1000; // Cooldown for 1s
      
      let newPhase: GameState = testShot.phase;
      let progress = 0;
      
      if (elapsed < trackingDuration) {
        // Tracking phase - ball moving fast
        newPhase = 'TRACKING';
        progress = elapsed / trackingDuration;
        // Use easeOutQuad for natural deceleration feel
        progress = 1 - (1 - progress) * (1 - progress);
        progress *= 0.3; // Only cover 30% of distance in tracking
      } else if (elapsed < trackingDuration + virtualDuration) {
        // Virtual rolling phase - ball continuing with deceleration
        newPhase = 'VIRTUAL_ROLLING';
        const virtualElapsed = elapsed - trackingDuration;
        progress = virtualElapsed / virtualDuration;
        // Stronger easeOut for rolling to a stop
        progress = 1 - Math.pow(1 - progress, 3);
        progress = 0.3 + progress * 0.7; // Cover remaining 70%
      } else if (elapsed < trackingDuration + virtualDuration + stopDuration) {
        // Stopped phase
        newPhase = 'STOPPED';
        progress = 1;
      } else if (elapsed < trackingDuration + virtualDuration + stopDuration + cooldownDuration) {
        // Cooldown phase
        newPhase = 'COOLDOWN';
        progress = 1;
      } else {
        // Done - return to armed
        setTestShot(null);
        return;
      }
      
      // Calculate current position
      const currentX = testShot.startX + (testShot.endX - testShot.startX) * progress;
      const currentY = testShot.startY + (testShot.endY - testShot.startY) * progress;
      
      setTestShot(prev => prev ? {
        ...prev,
        phase: newPhase,
        currentX,
        currentY,
      } : null);
      
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animationFrameRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [testShot?.active, testShot?.startTime]);

  // Keyboard listener for 'T' key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      
      if (e.key === 't' || e.key === 'T') {
        triggerTestShot();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [triggerTestShot]);

  // --- Merge test shot with real data ---
  const lastJsonMessage: BackendState | null = testShot?.active
    ? {
        // Use real data as base, override with test shot
        timestamp_ms: Date.now(),
        state: testShot.phase,
        lane: realLastJsonMessage?.lane || 'test',
        ball: testShot.phase === 'TRACKING' ? {
          x_px: testShot.currentX,
          y_px: testShot.currentY,
          radius_px: 20,
          confidence: 1.0,
        } : null,
        ball_visible: testShot.phase === 'TRACKING',
        velocity: {
          vx_px_s: (testShot.endX - testShot.startX) / (testShot.duration / 1000),
          vy_px_s: (testShot.endY - testShot.startY) / (testShot.duration / 1000),
          speed_px_s: testShot.shot.speed_px_s,
        },
        prediction: null,
        virtual_ball: (testShot.phase === 'VIRTUAL_ROLLING' || testShot.phase === 'STOPPED') ? {
          x: testShot.currentX,
          y: testShot.currentY,
          vx: 0,
          vy: 0,
          speed_m_s: testShot.phase === 'STOPPED' ? 0 : testShot.shot.speed_m_s * 0.3,
          distance_m: testShot.shot.distance_m,
          is_rolling: testShot.phase === 'VIRTUAL_ROLLING',
        } : null,
        shot: testShot.shot,
        metrics: realLastJsonMessage?.metrics || {
          cap_fps: 60,
          proc_fps: 60,
          disp_fps: 60,
          proc_latency_ms: 5,
          idle_stddev: 0.5,
        },
        calibrated: true,
        auto_calibrated: true,
        lens_calibrated: true,
        pixels_per_meter: realPixelsPerMeter,
        overlay_radius_scale: 1.0,
        resolution: realLastJsonMessage?.resolution || [1280, 800],
      }
    : realLastJsonMessage;

  // Convenience helpers
  const gameState = lastJsonMessage?.state || 'ARMED';
  const ballPosition = lastJsonMessage?.ball 
    ? { x: lastJsonMessage.ball.x_px, y: lastJsonMessage.ball.y_px } 
    : (lastJsonMessage?.virtual_ball ? { x: lastJsonMessage.virtual_ball.x, y: lastJsonMessage.virtual_ball.y } : null);
  const pixelsPerMeter = lastJsonMessage?.pixels_per_meter || 1150;
  
  // Game and session data
  const gameData = lastJsonMessage?.game || null;
  const sessionData = lastJsonMessage?.session || null;
  const drillData = lastJsonMessage?.drill || null;

  const sendReset = () => {
    // Also cancel test shot on reset
    if (testShot?.active) {
      setTestShot(null);
    }
    sendMessage(JSON.stringify({ type: 'reset' }));
  };
  
  // API functions for game management
  const setHoleDistance = useCallback(async (distance_m: number) => {
    try {
      const response = await fetch('http://localhost:8000/api/game/hole', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ distance_m })
      });
      if (!response.ok) {
        throw new Error('Failed to set hole distance');
      }
    } catch (error) {
      console.error('Error setting hole distance:', error);
    }
  }, []);
  
  const resetSession = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/session/reset', {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error('Failed to reset session');
      }
    } catch (error) {
      console.error('Error resetting session:', error);
    }
  }, []);
  
  const startDrill = useCallback(async (drillType: DrillType) => {
    try {
      const response = await fetch('http://localhost:8000/api/drill/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drill_type: drillType })
      });
      if (!response.ok) {
        throw new Error('Failed to start drill');
      }
    } catch (error) {
      console.error('Error starting drill:', error);
    }
  }, []);
  
  const stopDrill = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/drill/stop', {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error('Failed to stop drill');
      }
    } catch (error) {
      console.error('Error stopping drill:', error);
    }
  }, []);

  const value: WebSocketContextType = {
    readyState,
    lastJsonMessage,
    isConnected,
    gameState,
    ballPosition,
    pixelsPerMeter,
    sendReset,
    triggerTestShot,
    isTestShotActive: testShot?.active || false,
    // Game and session
    gameData,
    sessionData,
    drillData,
    setHoleDistance,
    resetSession,
    startDrill,
    stopDrill,
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
