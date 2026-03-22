import React, { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react';
import { ReadyState } from 'react-use-websocket';

import { useBackendHealth } from '../hooks/useBackendHealth';
import { useBackendSocket } from '../hooks/useBackendSocket';
import { useTestShotSimulation } from '../hooks/useTestShotSimulation';
import { useUiPreferences } from '../hooks/useUiPreferences';
import { useUsersApi } from '../hooks/useUsersApi';
import type {
  BackendState,
  DrillData,
  DrillType,
  GameState,
  GameStateData,
  MultiCameraState,
  SessionData,
  ShotReportData,
  ShotResult,
  ShotResultType,
  User,
} from '../types/backendState';

export type {
  BackendState,
  DrillData,
  DrillType,
  GameState,
  MultiCameraState,
  SessionData,
  ShotReportData,
  ShotResult,
  ShotResultType,
  User,
};
export type GameState_Data = GameStateData;

interface WebSocketContextType {
  readyState: ReadyState;
  lastJsonMessage: BackendState | null;
  isConnected: boolean;
  gameState: GameState;
  readyStatus: string;
  ballPosition: { x: number; y: number } | null;
  pixelsPerMeter: number;
  sendReset: () => void;
  triggerTestShot: () => void;
  isTestShotActive: boolean;
  gameData: GameStateData | null;
  sessionData: SessionData | null;
  drillData: DrillData | null;
  setHoleDistance: (distance_m: number) => Promise<void>;
  resetSession: () => Promise<void>;
  startDrill: (drillType: DrillType) => Promise<void>;
  stopDrill: () => Promise<void>;
  showDistance: boolean;
  setShowDistance: (show: boolean) => void;
  users: User[];
  refreshUsers: () => Promise<void>;
  createUser: (name: string, handicap: number) => Promise<void>;
  deleteUser: (userId: number) => Promise<void>;
  resetUserData: (userId: number) => Promise<{ success: boolean; shots_deleted?: number; sessions_deleted?: number }>;
  selectUser: (userId: number | null) => Promise<void>;
  multiCamera: MultiCameraState | null;
  shotReport: ShotReportData | null;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const { backendReady, httpBaseUrl } = useBackendHealth();
  const socket = useBackendSocket(backendReady);
  const uiPreferences = useUiPreferences();
  const testSimulation = useTestShotSimulation(
    socket.lastJsonMessage,
    socket.lastJsonMessage?.pixels_per_meter || 1150,
  );
  const usersApi = useUsersApi(httpBaseUrl, backendReady, socket.isConnected);

  const lastJsonMessage = testSimulation.lastJsonMessage;
  const gameState = lastJsonMessage?.state || 'ARMED';
  const readyStatus = lastJsonMessage?.ready_status || 'no_ball';
  const useVirtualPosition = !lastJsonMessage?.ball_visible || gameState === 'VIRTUAL_ROLLING' || gameState === 'STOPPED';
  const ballPosition = (useVirtualPosition && lastJsonMessage?.virtual_ball)
    ? { x: lastJsonMessage.virtual_ball.x, y: lastJsonMessage.virtual_ball.y }
    : (lastJsonMessage?.ball ? { x: lastJsonMessage.ball.x_px, y: lastJsonMessage.ball.y_px } : null);
  const pixelsPerMeter = lastJsonMessage?.pixels_per_meter || 1150;
  const gameData = lastJsonMessage?.game || null;
  const sessionData = lastJsonMessage?.session || null;
  const drillData = lastJsonMessage?.drill || null;
  const multiCamera = lastJsonMessage?.multi_camera || null;
  const shotReport = multiCamera?.shot_report || null;

  const postJson = useCallback(async (path: string, body?: unknown) => {
    try {
      const response = await fetch(`${httpBaseUrl}${path}`, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!response.ok) {
        throw new Error(`Request failed for ${path}`);
      }
    } catch (error) {
      console.error(`Error posting ${path}:`, error);
    }
  }, [httpBaseUrl]);

  const setHoleDistance = useCallback(async (distance_m: number) => {
    await postJson('/api/game/hole', { distance_m });
  }, [postJson]);

  const resetSession = useCallback(async () => {
    await postJson('/api/session/reset');
  }, [postJson]);

  const startDrill = useCallback(async (drillType: DrillType) => {
    await postJson('/api/drill/start', { drill_type: drillType });
  }, [postJson]);

  const stopDrill = useCallback(async () => {
    await postJson('/api/drill/stop');
  }, [postJson]);

  const sendReset = useCallback(() => {
    testSimulation.cancelTestShot();
    socket.sendReset();
  }, [socket, testSimulation]);

  const value = useMemo<WebSocketContextType>(() => ({
    readyState: socket.readyState,
    lastJsonMessage,
    isConnected: socket.isConnected,
    gameState,
    readyStatus,
    ballPosition,
    pixelsPerMeter,
    sendReset,
    triggerTestShot: testSimulation.triggerTestShot,
    isTestShotActive: testSimulation.isTestShotActive,
    gameData,
    sessionData,
    drillData,
    setHoleDistance,
    resetSession,
    startDrill,
    stopDrill,
    showDistance: uiPreferences.showDistance,
    setShowDistance: uiPreferences.setShowDistance,
    users: usersApi.users,
    refreshUsers: usersApi.refreshUsers,
    createUser: usersApi.createUser,
    deleteUser: usersApi.deleteUser,
    resetUserData: usersApi.resetUserData,
    selectUser: usersApi.selectUser,
    multiCamera,
    shotReport,
  }), [
    ballPosition,
    drillData,
    gameData,
    gameState,
    lastJsonMessage,
    multiCamera,
    pixelsPerMeter,
    readyStatus,
    sendReset,
    sessionData,
    setHoleDistance,
    resetSession,
    shotReport,
    socket.isConnected,
    socket.readyState,
    startDrill,
    stopDrill,
    testSimulation.isTestShotActive,
    testSimulation.triggerTestShot,
    uiPreferences.setShowDistance,
    uiPreferences.showDistance,
    usersApi.createUser,
    usersApi.deleteUser,
    usersApi.refreshUsers,
    usersApi.resetUserData,
    usersApi.selectUser,
    usersApi.users,
  ]);

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const usePuttingState = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('usePuttingState must be used within a WebSocketProvider');
  }
  return context;
};
