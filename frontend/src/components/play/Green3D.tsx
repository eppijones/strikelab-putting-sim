import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { PerspectiveCamera, Environment, OrbitControls } from '@react-three/drei';
import { VirtualGreenPlane } from './VirtualGreenPlane';
import { Ball3D } from './Ball3D';
import { AimLine3D } from './AimLine3D';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';
import type { CameraView } from './CameraSelector';

// import { SilkBackground } from './SilkBackground'; // Removed as per user request

// --- Scene Components ---

// DistanceIndicator removed as per user request

interface CameraProps {
  cameraView: CameraView;
}

// Standard follow camera - the default experience
const StandardCamera: React.FC = () => {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const trackingStartTimeRef = useRef<number | null>(null);
  const prevGameStateRef = useRef<string | null>(null);
  const lookAtTargetRef = useRef(new THREE.Vector3(0, 0, 5));
  const initialSpeedRef = useRef<number>(0);
  const { ballPosition, pixelsPerMeter, gameState, gameData, lastJsonMessage } = usePuttingState();

  // Camera configuration - DYNAMIC based on hole distance
  // Camera adjusts position to keep ball visible above UI elements
  const STATIC_DURATION = 0.5;
  
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  // Dynamic camera positioning based on hole distance
  // For longer putts, camera moves further back and adjusts height
  // This keeps the ball visible above the "READY FOR SHOT" text
  const BASE_HEIGHT = 1.8;
  const BASE_DISTANCE_BEHIND = 3.0;
  
  // Scale camera back for longer putts (starts scaling above 4m)
  const distanceScale = Math.max(0, (holeDistance - 4) * 0.4);
  const CAMERA_HEIGHT = BASE_HEIGHT + distanceScale * 0.3; // Slightly higher for long putts
  const CAMERA_DISTANCE_BEHIND = BASE_DISTANCE_BEHIND + distanceScale; // Further back for long putts
  
  // Start position - dynamically adjusted
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
      // ARMED state: smoothly adjust to current hole distance settings
      cameraRef.current.position.lerp(startPos, delta * 3);
      lookAtTargetRef.current.lerp(startLookAt, delta * 3);
      cameraRef.current.lookAt(lookAtTargetRef.current);
    }
    // STOPPED/COOLDOWN: camera stays where it stopped (do nothing)
  });

  return <PerspectiveCamera ref={cameraRef} makeDefault fov={50} position={[0, 1.8, -3.0]} />;
};

// Free Camera - user controlled with orbit controls
const FreeCamera: React.FC = () => {
  const { gameData } = usePuttingState();
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  const controlsRef = useRef<any>(null);
  
  return (
    <>
      <PerspectiveCamera makeDefault fov={50} position={[0, 3, -4]} />
      <OrbitControls
        ref={controlsRef}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={1}
        maxDistance={20}
        minPolarAngle={Math.PI * 0.1} // Limit how low you can go
        maxPolarAngle={Math.PI * 0.45} // Limit looking from below
        target={[0, 0, holeDistance / 2]}
        mouseButtons={{
          LEFT: THREE.MOUSE.ROTATE,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN,
        }}
        touches={{
          ONE: THREE.TOUCH.ROTATE,
          TWO: THREE.TOUCH.DOLLY_PAN,
        }}
        dampingFactor={0.05}
        enableDamping={true}
      />
    </>
  );
};

// Top-down view - bird's eye perspective
const TopDownCamera: React.FC = () => {
  const { gameData } = usePuttingState();
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  return (
    <>
      <PerspectiveCamera 
        makeDefault 
        fov={60} 
        position={[0, 12, holeDistance / 2]} 
        up={[0, 0, 1]} // Ensure hole is at the top (Z-up on screen)
      />
      <OrbitControls
        target={[0, 0, holeDistance / 2]}
        enableRotate={false}
        enableZoom={true}
        enablePan={true}
        minDistance={2}
        maxDistance={20}
        mouseButtons={{
          LEFT: THREE.MOUSE.PAN,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN,
        }}
        touches={{
          ONE: THREE.TOUCH.PAN,
          TWO: THREE.TOUCH.DOLLY_PAN,
        }}
      />
    </>
  );
};

// Cinematic camera - smooth automated movement
const CinematicCamera: React.FC = () => {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const { ballPosition, pixelsPerMeter, gameState, gameData } = usePuttingState();
  const timeRef = useRef(0);
  
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  useFrame((_, delta) => {
    if (!cameraRef.current) return;
    
    timeRef.current += delta * 0.3;
    
    const isTracking = gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING';
    
    if (isTracking && ballPosition) {
      // During tracking: dramatic side angle following the ball
      const ballZ = ballPosition.x / pixelsPerMeter;
      const angle = Math.sin(timeRef.current * 0.5) * 0.3;
      
      const camX = Math.sin(angle) * 4;
      const camZ = ballZ - 2;
      const camY = 1.5 + Math.sin(timeRef.current) * 0.3;
      
      cameraRef.current.position.lerp(new THREE.Vector3(camX, camY, camZ), delta * 2);
      cameraRef.current.lookAt(0, 0, Math.min(ballZ + 2, holeDistance));
    } else {
      // Idle: slow sweeping orbit around the scene
      const orbitRadius = 5;
      const orbitSpeed = 0.15;
      const camX = Math.sin(timeRef.current * orbitSpeed) * orbitRadius;
      const camZ = Math.cos(timeRef.current * orbitSpeed) * orbitRadius * 0.5;
      const camY = 2.5 + Math.sin(timeRef.current * 0.2) * 0.5;
      
      cameraRef.current.position.lerp(new THREE.Vector3(camX, camY, camZ), delta);
      cameraRef.current.lookAt(0, 0, holeDistance / 2);
    }
  });
  
  return <PerspectiveCamera ref={cameraRef} makeDefault fov={45} position={[4, 2.5, 0]} />;
};

// Camera rig that switches between modes
const CameraRig: React.FC<CameraProps> = ({ cameraView }) => {
  switch (cameraView) {
    case 'free':
      return <FreeCamera />;
    case 'top-down':
      return <TopDownCamera />;
    case 'cinematic':
      return <CinematicCamera />;
    case 'standard':
    default:
      return <StandardCamera />;
  }
};

interface Green3DProps {
  cameraView?: CameraView;
}

export const Green3D: React.FC<Green3DProps> = ({ cameraView = 'standard' }) => {
  return (
    <Canvas shadows className="absolute inset-0 z-0" gl={{ alpha: true }} resize={{ debounce: 0 }}>
      {/* 1. Environment & Lighting - Nordic Light Theme */}
      <color attach="background" args={['#F2F0EB']} /> {/* Nordic Paper Background */}
      
      <fog attach="fog" args={['#F2F0EB', 10, 40]} /> {/* Match background base color */}
      
      {/* Soft, Bright Studio Lighting */}
      <ambientLight intensity={0.8} color="#ffffff" /> {/* Increased intensity for light theme */}
      <directionalLight 
        position={[5, 10, 5]} 
        intensity={0.8} // Brighter key light
        castShadow 
        shadow-mapSize={[1024, 1024]}
        shadow-bias={-0.0001}
      />
      {/* Subtle environment reflection */}
      <Environment preset="city" blur={1} /> {/* City has warmer tones than studio */}

      {/* 2. Camera Rig */}
      <CameraRig cameraView={cameraView} />

      {/* 3. Objects */}
      <VirtualGreenPlane />
      <AimLine3D />
      <Ball3D />
      {/* <DistanceIndicator /> removed */}
      
      {/* 4. Floor - Clean minimalist plane matching background */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#F2F0EB" roughness={1} />
      </mesh>

    </Canvas>
  );
};
