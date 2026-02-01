import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

export const Ball3D: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { ballPosition, pixelsPerMeter } = usePuttingState();

  useFrame((_, delta) => {
    if (!meshRef.current) return;

    if (ballPosition) {
      // Coordinate Mapping (Same as Camera)
      // Z = Forward (Putt direction) = X pixel
      // X = Horizontal (Left/Right) = Y pixel (centered)
      
      const zMeters = ballPosition.x / pixelsPerMeter;
      const xMeters = (ballPosition.y - 400) / pixelsPerMeter; // Center 400 assuming 800w? No, backend sends frame coords.
      // If frame is 800px high (vertical video? or landscape?)
      // Standard: 1280x800.
      // So X is 0..1280 (Width), Y is 0..800 (Height).
      // Wait, 2D visualizer logic was:
      // transformX(camY) -> Screen X.
      // transformY(camX) -> Screen Y.
      // So Y is Left/Right. X is Distance.
      
      // Correct Mapping:
      // 3D Z (Forward) = ballPosition.x / PPM
      // 3D X (Left/Right) = -(ballPosition.y - (800/2)) / PPM  (Note negative for Right-Hand Coord System?)
      // Let's assume standard ThreeJS:
      // X+ Right, Y+ Up, Z+ Back (towards camera).
      // So Forward is Z- ?
      // Our camera is at Z = -1.5, looking at Z = 0.
      // So Forward is Z+.
      
      const targetPos = new THREE.Vector3(xMeters, 0.0213, zMeters); // Radius 0.0213 (42.67mm / 2)
      
      // Smooth interpolation for visual jitter reduction
      meshRef.current.position.lerp(targetPos, delta * 20);
      
      // Rotation (fake rolling based on velocity would be better, but simple roll for now)
      // We need velocity from state to do this properly.
      // For now, just slide.
    }
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
