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
      // ALWAYS CENTER: Ball always appears at X=0 (center of screen)
      // regardless of physical position on the mat
      // Only Z (distance/forward) position is used from tracking
      // This creates a "virtual straight putt" view
      
      const zMeters = ballPosition.x / pixelsPerMeter;
      // X is always 0 - ball stays centered on screen
      
      targetPos = new THREE.Vector3(0, 0.0213, zMeters); // Always X=0, only Z changes
      
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
