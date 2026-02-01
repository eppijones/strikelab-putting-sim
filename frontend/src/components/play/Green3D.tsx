import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera, Environment, Html } from '@react-three/drei';
import { VirtualGreenPlane } from './VirtualGreenPlane';
import { Ball3D } from './Ball3D';
import { AimLine3D } from './AimLine3D';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

// --- Scene Components ---

const DistanceIndicator: React.FC = () => {
  const { ballPosition, pixelsPerMeter, gameData, showDistance } = usePuttingState();

  if (!showDistance) return null;

  // Ball position in meters (relative to start)
  const ballZ = ballPosition ? ballPosition.x / pixelsPerMeter : 0;
  
  // Hole position (fixed at gameData distance)
  const holeZ = gameData?.hole?.distance_m ?? 3;
  
  // Calculate remaining distance
  const remainingDistM = Math.max(0, holeZ - ballZ);
  const remainingDistCm = Math.round(remainingDistM * 100);

  // Position label at midpoint, slightly elevated
  // Ensure it doesn't jump around too much when ball is moving
  const midZ = ballZ + (holeZ - ballZ) / 2;
  
  // Don't show if ball passed hole
  if (ballZ > holeZ) return null;

  return (
    <Html position={[0, 0.2, midZ]} center zIndexRange={[100, 0]}>
      <div className="px-3 py-1 bg-white/90 backdrop-blur-sm rounded-full shadow-lg border border-white/20 select-none">
        <span className="text-sm font-bold text-slate-700 font-mono whitespace-nowrap">
          {remainingDistCm} cm
        </span>
      </div>
    </Html>
  );
};

const DynamicCamera: React.FC = () => {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const trackingStartTimeRef = useRef<number | null>(null);
  const prevGameStateRef = useRef<string | null>(null);
  const lookAtTargetRef = useRef(new THREE.Vector3(0, 0, 5));
  const initialSpeedRef = useRef<number>(0);
  const { ballPosition, pixelsPerMeter, gameState, gameData, lastJsonMessage } = usePuttingState();

  // Camera configuration - FIXED CENTERED VIEW
  // Camera stays at X=0, looking straight at the hole
  // Ball appears wherever it is, aim line shows its path
  const STATIC_DURATION = 0.5;
  const CAMERA_HEIGHT = 2.5;
  const CAMERA_DISTANCE_BEHIND = 3.0; // Further back for better view
  
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  // Fixed start position - always centered, looking straight at hole
  const startPos = new THREE.Vector3(0, CAMERA_HEIGHT, -CAMERA_DISTANCE_BEHIND);
  const startLookAt = new THREE.Vector3(0, 0, holeDistance);

  useFrame((_, delta) => {
    if (!cameraRef.current) return;

    const isTracking = gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING';
    const wasTracking = prevGameStateRef.current === 'TRACKING' || prevGameStateRef.current === 'VIRTUAL_ROLLING';
    const isArmed = gameState === 'ARMED';
    const wasArmed = prevGameStateRef.current === 'ARMED';

    // Get current ball position in meters (for follow during tracking)
    let ballZ = 0;
    if (ballPosition) {
      ballZ = ballPosition.x / pixelsPerMeter;
    }

    if (isTracking && !wasTracking) {
      trackingStartTimeRef.current = performance.now() / 1000;
      const speed = lastJsonMessage?.velocity?.speed_px_s || 
                    (lastJsonMessage?.virtual_ball?.speed_m_s ? lastJsonMessage.virtual_ball.speed_m_s * pixelsPerMeter : 1500);
      initialSpeedRef.current = Math.max(speed, 500);
    }

    // Snap camera when transitioning to ARMED
    if (isArmed && !wasArmed) {
      trackingStartTimeRef.current = null;
      cameraRef.current.position.copy(startPos);
      lookAtTargetRef.current.copy(startLookAt);
      cameraRef.current.lookAt(lookAtTargetRef.current);
    }

    prevGameStateRef.current = gameState;

    if (isTracking && trackingStartTimeRef.current !== null) {
      const elapsed = (performance.now() / 1000) - trackingStartTimeRef.current;

      // During tracking: camera moves forward (Z only), stays centered (X=0)
      // ALWAYS stay behind the ball - never pass it
      // Camera Z position = ball Z - full distance behind (same as start distance)
      const cameraZ = ballZ - CAMERA_DISTANCE_BEHIND;
      const followPos = new THREE.Vector3(0, CAMERA_HEIGHT * 0.9, cameraZ);
      const followLookAt = new THREE.Vector3(0, 0, Math.max(ballZ + 2, holeDistance));

      let currentSpeed = 0;
      if (lastJsonMessage?.velocity?.speed_px_s) {
        currentSpeed = lastJsonMessage.velocity.speed_px_s;
      } else if (lastJsonMessage?.virtual_ball?.speed_m_s) {
        currentSpeed = lastJsonMessage.virtual_ball.speed_m_s * pixelsPerMeter;
      }
      
      const speedRatio = Math.max(0.08, currentSpeed / initialSpeedRef.current);
      const speedFactor = Math.sqrt(speedRatio);

      if (elapsed < STATIC_DURATION) {
        // Brief pause at start
        cameraRef.current.lookAt(startLookAt);
      } else {
        const lerpSpeed = delta * 1.5 * speedFactor;
        cameraRef.current.position.lerp(followPos, lerpSpeed);
        lookAtTargetRef.current.lerp(followLookAt, lerpSpeed);
        cameraRef.current.lookAt(lookAtTargetRef.current);
      }
    } else if (isArmed) {
      // ARMED state: camera stays fixed centered
      cameraRef.current.lookAt(startLookAt);
    }
    // STOPPED/COOLDOWN: camera stays where it stopped (do nothing)
  });

  return <PerspectiveCamera ref={cameraRef} makeDefault fov={50} position={[0, 2.5, -3.0]} />;
};

export const Green3D: React.FC = () => {
  return (
    <Canvas shadows className="absolute inset-0 z-0">
      {/* 1. Environment & Lighting - Cozy Scandinavian Minimalist */}
      <color attach="background" args={['#E5E0D8']} /> {/* Warm Grey/Greige background - less bright */}
      <fog attach="fog" args={['#E5E0D8', 10, 40]} /> {/* Fade to match background */}
      
      {/* Soft, Warm Studio Lighting */}
      <ambientLight intensity={0.5} color="#ffffff" /> {/* Reduced intensity */}
      <directionalLight 
        position={[5, 10, 5]} 
        intensity={0.6} // Softer key light
        castShadow 
        shadow-mapSize={[1024, 1024]}
        shadow-bias={-0.0001}
      />
      {/* Subtle environment reflection */}
      <Environment preset="city" blur={1} /> {/* City has warmer tones than studio */}

      {/* 2. Camera Rig */}
      <DynamicCamera />

      {/* 3. Objects */}
      <VirtualGreenPlane />
      <AimLine3D />
      <Ball3D />
      <DistanceIndicator />
      
      {/* 4. Floor - Clean minimalist plane matching background */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#E5E0D8" roughness={1} />
      </mesh>

    </Canvas>
  );
};
