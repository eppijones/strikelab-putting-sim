import { useCallback, useEffect, useRef, useState } from 'react';

import type { BackendState, GameState, ShotResult } from '../types/backendState';

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
  duration: number;
}

function generateTestShot(pixelsPerMeter: number): TestShotState {
  const speed_m_s = 1.5 + Math.random() * 2.5;
  const direction_deg = (Math.random() - 0.5) * 10;
  const distance_m = 2 + Math.random() * 6;
  const speed_px_s = speed_m_s * pixelsPerMeter;
  const distance_px = distance_m * pixelsPerMeter;
  const startX = 100;
  const startY = 400;
  const direction_rad = (direction_deg * Math.PI) / 180;
  const endX = startX + distance_px * Math.cos(direction_rad);
  const endY = startY + distance_px * Math.sin(direction_rad);

  const trajectory: number[][] = [];
  for (let i = 0; i <= 50; i += 1) {
    const t = i / 50;
    trajectory.push([
      startX + (endX - startX) * t,
      startY + (endY - startY) * t,
    ]);
  }

  return {
    active: true,
    startTime: Date.now(),
    phase: 'TRACKING',
    shot: {
      speed_m_s,
      speed_px_s,
      direction_deg,
      physical_distance_m: distance_m * 0.3,
      virtual_distance_m: distance_m * 0.7,
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
    duration: (distance_m / speed_m_s) * 2000,
  };
}

export function useTestShotSimulation(realLastJsonMessage: BackendState | null, realPixelsPerMeter: number) {
  const [testShot, setTestShot] = useState<TestShotState | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const triggerTestShot = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setTestShot(generateTestShot(realPixelsPerMeter));
  }, [realPixelsPerMeter]);

  useEffect(() => {
    if (!testShot?.active) {
      return undefined;
    }

    const animate = () => {
      const elapsed = Date.now() - testShot.startTime;
      const totalDistance = testShot.shot.distance_px;
      const initialSpeed = testShot.shot.speed_px_s;
      const deceleration = (initialSpeed * initialSpeed) / (2 * totalDistance);
      const stopTimeMs = (2 * totalDistance / initialSpeed) * 1000;
      const trackingDistance = totalDistance * 0.25;
      const trackingTimeMs = (
        initialSpeed - Math.sqrt(initialSpeed * initialSpeed - 2 * deceleration * trackingDistance)
      ) / deceleration * 1000;
      const virtualTimeMs = stopTimeMs - trackingTimeMs;

      let newPhase: GameState = testShot.phase;
      let distance = 0;
      if (elapsed < trackingTimeMs) {
        newPhase = 'TRACKING';
        const t = elapsed / 1000;
        distance = initialSpeed * t - 0.5 * deceleration * t * t;
      } else if (elapsed < trackingTimeMs + virtualTimeMs) {
        newPhase = 'VIRTUAL_ROLLING';
        const t = elapsed / 1000;
        distance = Math.min(initialSpeed * t - 0.5 * deceleration * t * t, totalDistance);
      } else if (elapsed < trackingTimeMs + virtualTimeMs + 2000) {
        newPhase = 'STOPPED';
        distance = totalDistance;
      } else if (elapsed < trackingTimeMs + virtualTimeMs + 3000) {
        newPhase = 'COOLDOWN';
        distance = totalDistance;
      } else {
        setTestShot(null);
        return;
      }

      const progress = distance / totalDistance;
      setTestShot((previous) => previous ? {
        ...previous,
        phase: newPhase,
        currentX: previous.startX + (previous.endX - previous.startX) * progress,
        currentY: previous.startY + (previous.endY - previous.startY) * progress,
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

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (event.key === 't' || event.key === 'T') {
        triggerTestShot();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [triggerTestShot]);

  const mergedState: BackendState | null = testShot?.active
    ? {
        timestamp_ms: Date.now(),
        state: testShot.phase,
        lane: realLastJsonMessage?.lane || 'test',
        ball: testShot.phase === 'TRACKING'
          ? {
              x_px: testShot.currentX,
              y_px: testShot.currentY,
              radius_px: 20,
              confidence: 1,
            }
          : null,
        ball_visible: testShot.phase === 'TRACKING',
        velocity: {
          vx_px_s: (testShot.endX - testShot.startX) / (testShot.duration / 1000),
          vy_px_s: (testShot.endY - testShot.startY) / (testShot.duration / 1000),
          speed_px_s: testShot.shot.speed_px_s,
        },
        prediction: null,
        virtual_ball: (testShot.phase === 'VIRTUAL_ROLLING' || testShot.phase === 'STOPPED')
          ? {
              x: testShot.currentX,
              y: testShot.currentY,
              vx: 0,
              vy: 0,
              speed_m_s: testShot.phase === 'STOPPED' ? 0 : testShot.shot.speed_m_s * 0.3,
              distance_m: testShot.shot.distance_m,
              is_rolling: testShot.phase === 'VIRTUAL_ROLLING',
            }
          : null,
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
        overlay_radius_scale: 1,
        resolution: realLastJsonMessage?.resolution || [1280, 800],
        ready_status: realLastJsonMessage?.ready_status || 'ready',
        game: realLastJsonMessage?.game || null,
        session: realLastJsonMessage?.session || null,
        drill: realLastJsonMessage?.drill || null,
        multi_camera: realLastJsonMessage?.multi_camera || null,
      }
    : realLastJsonMessage;

  const cancelTestShot = useCallback(() => {
    setTestShot(null);
  }, []);

  return {
    lastJsonMessage: mergedState,
    triggerTestShot,
    isTestShotActive: testShot?.active || false,
    cancelTestShot,
  };
}
