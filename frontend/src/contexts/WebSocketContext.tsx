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

export interface ConsistencyMetrics {
  speed_stddev: number;
  direction_stddev: number;
  distance_error_stddev: number;
  speed_cv: number;
  consistency_score: number;
  rolling_speed_stddev: number;
  rolling_direction_stddev: number;
}

export interface TendencyAnalysis {
  speed_bias_m_s: number;
  distance_bias_m: number;
  direction_bias_deg: number;
  lateral_bias_m: number;
  dominant_miss: string;
  dominant_miss_percentage: number;
  speed_tendency: string;
  direction_tendency: string;
}

export interface MissDistribution {
  right_short: number;
  right_long: number;
  left_short: number;
  left_long: number;
  right_short_pct: number;
  right_long_pct: number;
  left_short_pct: number;
  left_long_pct: number;
  total_right: number;
  total_left: number;
  total_short: number;
  total_long: number;
  total_misses: number;
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
  // NEW: Consistency and analytics
  consistency: ConsistencyMetrics;
  tendencies: TendencyAnalysis;
  miss_distribution: MissDistribution;
  user_id?: number | null;
}

export interface User {
  id: number;
  name: string;
  handicap: number;
  created_at: string;
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
  showDistance: boolean;
  setShowDistance: (show: boolean) => void;
  // User Management
  users: User[];
  refreshUsers: () => Promise<void>;
  createUser: (name: string, handicap: number) => Promise<void>;
  deleteUser: (userId: number) => Promise<void>;
  resetUserData: (userId: number) => Promise<{ success: boolean; shots_deleted?: number; sessions_deleted?: number }>;
  selectUser: (userId: number | null) => Promise<void>;
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

  // --- UI State ---
  const [showDistance, setShowDistance] = useState(true);
  
  // --- User State ---
  const [users, setUsers] = useState<User[]>([]);

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

  // Test shot animation loop - uses LINEAR DECELERATION physics (same as backend)
  // v(t) = v0 - a*t, where a = v0²/(2*d) for stopping at distance d
  // x(t) = v0*t - 0.5*a*t² 
  // Stops at t_stop = v0/a = 2*d/v0
  useEffect(() => {
    if (!testShot?.active) return;

    const animate = () => {
      const elapsed = Date.now() - testShot.startTime;
      
      // Physics parameters
      const totalDistance = testShot.shot.distance_px;
      const initialSpeed = testShot.shot.speed_px_s;
      
      // Linear deceleration: a = v0²/(2*d)
      const deceleration = (initialSpeed * initialSpeed) / (2 * totalDistance);
      
      // Time to stop: t_stop = v0/a = 2*d/v0
      const stopTime_ms = (2 * totalDistance / initialSpeed) * 1000;
      
      // Phases
      const trackingDistance = totalDistance * 0.25; // Ball in frame for 25% of distance
      const trackingTime_ms = (initialSpeed - Math.sqrt(initialSpeed * initialSpeed - 2 * deceleration * trackingDistance)) / deceleration * 1000;
      const virtualTime_ms = stopTime_ms - trackingTime_ms;
      const stopDuration = 2000;
      const cooldownDuration = 1000;
      
      let newPhase: GameState = testShot.phase;
      let distance = 0;
      
      if (elapsed < trackingTime_ms) {
        // Tracking phase - ball decelerating in frame
        newPhase = 'TRACKING';
        const t = elapsed / 1000;
        // x(t) = v0*t - 0.5*a*t²
        distance = initialSpeed * t - 0.5 * deceleration * t * t;
      } else if (elapsed < trackingTime_ms + virtualTime_ms) {
        // Virtual rolling phase - continuing deceleration
        newPhase = 'VIRTUAL_ROLLING';
        const t = elapsed / 1000;
        // Same physics formula, just continuing
        distance = initialSpeed * t - 0.5 * deceleration * t * t;
        // Clamp to total distance
        distance = Math.min(distance, totalDistance);
      } else if (elapsed < trackingTime_ms + virtualTime_ms + stopDuration) {
        // Stopped phase
        newPhase = 'STOPPED';
        distance = totalDistance;
      } else if (elapsed < trackingTime_ms + virtualTime_ms + stopDuration + cooldownDuration) {
        // Cooldown phase
        newPhase = 'COOLDOWN';
        distance = totalDistance;
      } else {
        // Done - return to armed
        setTestShot(null);
        return;
      }
      
      // Calculate current position from distance traveled
      const progress = distance / totalDistance;
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

  // --- User Management API ---
  const refreshUsers = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/users');
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setUsers(data.users);
        }
      }
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  }, []);

  const createUser = useCallback(async (name: string, handicap: number) => {
    try {
      const response = await fetch('http://localhost:8000/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, handicap })
      });
      if (response.ok) {
        await refreshUsers();
      }
    } catch (error) {
      console.error('Error creating user:', error);
    }
  }, [refreshUsers]);

  const deleteUser = useCallback(async (userId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/users/${userId}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        await refreshUsers();
      }
    } catch (error) {
      console.error('Error deleting user:', error);
    }
  }, [refreshUsers]);

  const resetUserData = useCallback(async (userId: number): Promise<{ success: boolean; shots_deleted?: number; sessions_deleted?: number }> => {
    try {
      const response = await fetch(`http://localhost:8000/api/users/${userId}/reset`, {
        method: 'POST'
      });
      if (response.ok) {
        const data = await response.json();
        return { 
          success: true, 
          shots_deleted: data.shots_deleted, 
          sessions_deleted: data.sessions_deleted 
        };
      }
      return { success: false };
    } catch (error) {
      console.error('Error resetting user data:', error);
      return { success: false };
    }
  }, []);

  const selectUser = useCallback(async (userId: number | null) => {
    try {
      const response = await fetch('http://localhost:8000/api/session/user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
      });
      if (!response.ok) {
        throw new Error('Failed to select user');
      }
    } catch (error) {
      console.error('Error selecting user:', error);
    }
  }, []);

  // Initial load of users
  useEffect(() => {
    refreshUsers();
  }, [refreshUsers]);

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
    showDistance,
    setShowDistance,
    // User Management
    users,
    refreshUsers,
    createUser,
    deleteUser,
    resetUserData,
    selectUser,
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
