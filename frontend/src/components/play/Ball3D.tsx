import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

export const Ball3D: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { ballPosition, pixelsPerMeter, gameState } = usePuttingState();

  useFrame((_, delta) => {
    if (!meshRef.current) return;

    let targetPos: THREE.Vector3 | null = null;

    if (ballPosition) {
      // Coordinate Mapping (Same as Camera)
      // Z = Forward (Putt direction) = X pixel
      // X = Horizontal (Left/Right) = Y pixel (centered)
      
      const zMeters = ballPosition.x / pixelsPerMeter;
      const xMeters = (ballPosition.y - 400) / pixelsPerMeter; // Center 400 assuming 800w
      
      // Correct Mapping:
      // 3D Z (Forward) = ballPosition.x / PPM
      // 3D X (Left/Right) = -(ballPosition.y - (800/2)) / PPM
      // We assume camera at Z=-1.5 looking at Z=0.
      
      targetPos = new THREE.Vector3(xMeters, 0.0213, zMeters); // Radius 0.0213 (42.67mm / 2)
      
      // Smooth interpolation for visual jitter reduction
      meshRef.current.position.lerp(targetPos, delta * 20);
    } else if (gameState === 'ARMED') {
      // Reset to origin (0,0) which is centered to the flag line
      targetPos = new THREE.Vector3(0, 0.0213, 0);
      meshRef.current.position.lerp(targetPos, delta * 5);
    }
      
    // Rotation (fake rolling based on velocity would be better, but simple roll for now)
    // We need velocity from state to do this properly.
    // For now, just slide.
  });

  return (
    <mesh ref={meshRef} position={[0, 0.0213, 0]} castShadow>
      <sphereGeometry args={[0.0213, 32, 32]} />
      <meshStandardMaterial 
        color="#ffffff" 
        roughness={0.3} 
        metalness={0.1}
      />
    </mesh>
  );
};
