import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera, Environment } from '@react-three/drei';
import { VirtualGreenPlane } from './VirtualGreenPlane';
import { Ball3D } from './Ball3D';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

// --- Scene Components ---

const DynamicCamera: React.FC = () => {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const { ballPosition, pixelsPerMeter, gameState } = usePuttingState();

  // Mapping: Pixel -> Meter (See Plan)
  // Backend X (pixels) -> 3D Z (Meters Forward)
  // Backend Y (pixels) -> 3D X (Meters Left/Right)
  // Origin offset needs to be calibrated. For now, assume 0,0 pixel is camera start.
  
  // Actually, let's look at GreenVisualizer 2D mapping:
  // camHeight/2 is center Y (Screen X).
  // camWidth is top X (Screen Y).
  // 3D coordinate system: Z- is forward (standard OpenGL), or Z+ is forward?
  // Let's use:
  // Z = Forward (Putt direction)
  // X = Horizontal (Left/Right)
  // Y = Up (Vertical)

  useFrame((_, delta) => {
    if (!cameraRef.current) return;

    // Target position (Ball or Start)
    let targetZ = 0;
    let targetX = 0;

    if (ballPosition) {
        // Convert pixels to meters (approximate origin adjustment)
        // Assuming X pixel increases as ball moves away from camera
        const zMeters = ballPosition.x / pixelsPerMeter; 
        
        // Assuming Y pixel 400 is center (800w resolution? no 1280x800)
        // If resolution is 1280x800, center Y is 400.
        // Y pixel increases to the right? Or down? OpenCV usually Y is down.
        // Let's assume standard image coords: 0,0 top-left.
        // If camera is mounted overhead:
        // We need to know orientation. 
        // Let's stick to the 2D visualizer logic:
        // transformX uses (camY - center) -> Screen X.
        
        // So:
        const xMeters = (ballPosition.y - 400) / pixelsPerMeter; // 400 is half of 800 height
        
        targetZ = zMeters;
        targetX = xMeters;
    }

    // Camera State Logic
    // Start: Behind the tee
    const startPos = new THREE.Vector3(0, 1.2, -1.5); // 1.2m up, 1.5m behind
    const lookAtOffset = new THREE.Vector3(0, 0, 2); // Look 2m ahead

    if (gameState === 'TRACKING' || gameState === 'VIRTUAL_ROLLING') {
       // Follow mode
       const followPos = new THREE.Vector3(targetX, 0.8, targetZ - 1.0);
       // Smooth lerp
       cameraRef.current.position.lerp(followPos, delta * 2);
       
       const targetLook = new THREE.Vector3(targetX, 0, targetZ + 3);
       // We can't lerp lookAt directly on the camera object easily without OrbitControls,
       // but we can lerp a dummy target vector.
       // For MVP, just updating position and lookAt frame-by-frame is okay.
       cameraRef.current.lookAt(targetLook);
    } else {
       // Reset / Idle
       cameraRef.current.position.lerp(startPos, delta * 2);
       cameraRef.current.lookAt(lookAtOffset);
    }
  });

  return <PerspectiveCamera ref={cameraRef} makeDefault fov={50} position={[0, 1.2, -1.5]} />;
};

export const Green3D: React.FC = () => {
  return (
    <Canvas shadows className="absolute inset-0 z-0">
      {/* 1. Environment & Lighting */}
      <color attach="background" args={['#242322']} /> {/* midnight-surface */}
      <fog attach="fog" args={['#242322', 2, 15]} /> {/* Fade into void */}
      
      {/* SoftShadows removed due to shader compilation errors with Three.js r160+ */}
      <ambientLight intensity={0.6} />
      <directionalLight 
        position={[5, 10, -5]} 
        intensity={1} 
        castShadow 
        shadow-bias={-0.0001}
      />
      <Environment preset="studio" blur={0.8} />

      {/* 2. Camera Rig */}
      <DynamicCamera />

      {/* 3. Objects */}
      <VirtualGreenPlane />
      <Ball3D />
      
      {/* 4. Floor (Void catcher) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#1E3A2B" roughness={1} />
      </mesh>

    </Canvas>
  );
};
