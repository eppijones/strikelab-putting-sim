import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

export const Ball3D: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { ballPosition, pixelsPerMeter, gameState, lastJsonMessage, gameData } = usePuttingState();
  
  const [isDropped, setIsDropped] = useState(false);
  const dropTimeRef = useRef<number | null>(null);
  
  const startOffsetXRef = useRef<number | null>(null);
  const startOffsetZRef = useRef<number | null>(null);
  const prevGameStateRef = useRef<string | null>(null);
  
  const frozenPositionRef = useRef<THREE.Vector3 | null>(null);
  
  useEffect(() => {
    if (gameState === 'ARMED') {
      setIsDropped(false);
      dropTimeRef.current = null;
      startOffsetXRef.current = null;
      startOffsetZRef.current = null;
      frozenPositionRef.current = null;
    }
  }, [gameState]);

  useFrame((state, delta) => {
    if (!meshRef.current) return;

    const holeDistance = gameData?.hole?.distance_m ?? 3;
    const holeRadius = 0.054;
    const dropDepth = -0.1;
    const ballRadius = 0.0213;
    const MAX_DROP_SPEED = 1.5;

    if (gameState === 'STOPPED' && !frozenPositionRef.current && !isDropped) {
      frozenPositionRef.current = meshRef.current.position.clone();
    }
    
    if (gameState === 'STOPPED' && frozenPositionRef.current && !isDropped) {
      meshRef.current.position.copy(frozenPositionRef.current);
      prevGameStateRef.current = gameState;
      return;
    }

    let trackerPos = new THREE.Vector3(0, ballRadius, 0);
    let trackerSpeed = 0;

    const shotFinalDistance = lastJsonMessage?.shot?.distance_m;

    if (ballPosition) {
      const rawZMeters = ballPosition.x / pixelsPerMeter;
      
      const resolution = lastJsonMessage?.resolution || [1280, 800];
      const centerY = resolution[1] / 2;
      const rawXMeters = (centerY - ballPosition.y) / pixelsPerMeter;
      
      if (gameState === 'ARMED') {
        startOffsetXRef.current = rawXMeters;
        startOffsetZRef.current = rawZMeters;
      } else if (startOffsetXRef.current === null || startOffsetZRef.current === null) {
        startOffsetXRef.current = rawXMeters;
        startOffsetZRef.current = rawZMeters;
      }
      
      const normalizedX = rawXMeters - (startOffsetXRef.current ?? 0);
      const normalizedZ = rawZMeters - (startOffsetZRef.current ?? 0);
      
      const clampedX = Math.max(-2, Math.min(2, normalizedX));
      trackerPos.set(clampedX, ballRadius, normalizedZ);
      
      if (lastJsonMessage?.virtual_ball?.speed_m_s != null) {
        trackerSpeed = lastJsonMessage.virtual_ball.speed_m_s;
      } else if (lastJsonMessage?.velocity?.speed_px_s) {
        trackerSpeed = lastJsonMessage.velocity.speed_px_s / pixelsPerMeter;
      }
    } else if (gameState === 'ARMED') {
       meshRef.current.position.lerp(new THREE.Vector3(0, ballRadius, 0), delta * 5);
       meshRef.current.rotation.set(0, 0, 0);
       prevGameStateRef.current = gameState;
       return;
    }
    
    prevGameStateRef.current = gameState;

    // --- Simple hole interaction: drop in or roll over ---
    const dx = trackerPos.x;
    const dz = trackerPos.z - holeDistance;
    const distToHoleCenter = Math.sqrt(dx * dx + dz * dz);

    if (!isDropped && distToHoleCenter < holeRadius && trackerSpeed < MAX_DROP_SPEED && trackerSpeed > 0.01) {
      setIsDropped(true);
      dropTimeRef.current = state.clock.elapsedTime;
    }

    let finalPos = trackerPos;

    if (isDropped && dropTimeRef.current) {
      const timeInHole = state.clock.elapsedTime - dropTimeRef.current;
      const dropProgress = Math.min(timeInHole * 3, 1);
      finalPos = new THREE.Vector3(
        THREE.MathUtils.lerp(trackerPos.x, 0, dropProgress * 0.5),
        THREE.MathUtils.lerp(ballRadius, dropDepth, dropProgress),
        THREE.MathUtils.lerp(trackerPos.z, holeDistance, dropProgress * 0.5),
      );
    } else {
      if (shotFinalDistance && (gameState === 'VIRTUAL_ROLLING' || gameState === 'STOPPED')) {
        trackerPos.z = Math.min(trackerPos.z, shotFinalDistance);
      }
      finalPos = trackerPos;
    }

    meshRef.current.position.lerp(finalPos, delta * 25);

    if (trackerSpeed > 0.05) {
      meshRef.current.rotation.x += trackerSpeed * delta * 5; 
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
