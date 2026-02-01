import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

/**
 * AimLine3D - Ghost shot / aiming line visualization
 * Shows:
 * 1. Target line from ball to hole
 * 2. Animated rolling dots to indicate ideal tempo
 * 3. Only visible when ARMED (ready for shot)
 */

// Animated dot that travels along the aim line
const RollingDot: React.FC<{ 
  holeDistance: number; 
  delay: number; 
  speed: number;
}> = ({ holeDistance, delay, speed }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const progressRef = useRef(delay);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    
    // Animate progress along the line
    progressRef.current += delta * speed;
    
    // Loop when reaching the end (small pause at end before restart)
    if (progressRef.current > 1.15) {
      progressRef.current = 0;
    }
    
    // Clamp for position calculation
    const t = Math.min(progressRef.current, 1);
    
    // Realistic deceleration: ball starts fast, slows down as it rolls
    // Using ease-out curve: 1 - (1-t)^2
    const easedProgress = 1 - Math.pow(1 - t, 2);
    
    // Calculate position along the line (start to hole)
    const z = easedProgress * holeDistance;
    meshRef.current.position.z = z;
    
    // Fade in quickly at start, fade out at end
    const fadeIn = Math.min(progressRef.current * 8, 1);
    const fadeOut = progressRef.current > 1 ? Math.max(0, 1 - (progressRef.current - 1) * 6) : 1;
    const opacity = fadeIn * fadeOut * 0.6;
    
    const material = meshRef.current.material as THREE.MeshBasicMaterial;
    material.opacity = opacity;
  });

  return (
    <mesh ref={meshRef} position={[0, 0.008, 0]}>
      <sphereGeometry args={[0.018, 16, 16]} />
      <meshBasicMaterial 
        color="#8EB897" 
        transparent 
        opacity={0.5}
      />
    </mesh>
  );
};

export const AimLine3D: React.FC = () => {
  const { gameState, gameData } = usePuttingState();
  const groupRef = useRef<THREE.Group>(null);
  const fadeRef = useRef(0);
  
  // Get hole distance from game data, default to 3m
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  // Calculate realistic putting speed based on physics
  // Initial velocity needed: v = sqrt(2 * friction * g * distance)
  // Simplified: ~1.4 m/s for 3m, ~2.0 m/s for 6m, etc.
  const initialSpeed = Math.sqrt(holeDistance * 0.65); // m/s
  
  // Time for ball to roll the distance (accounting for deceleration)
  // t = 2 * distance / initial_speed (due to linear deceleration to 0)
  const rollTime = (2 * holeDistance) / initialSpeed; // seconds
  
  // Convert to animation speed (progress per second = 1 / rollTime)
  const dotSpeed = 1 / rollTime;
  
  // Create the dashed line geometry
  const lineGeometry = useMemo(() => {
    const points = [];
    const segments = 100;
    
    for (let i = 0; i <= segments; i++) {
      const z = (i / segments) * holeDistance;
      points.push(new THREE.Vector3(0, 0.003, z));
    }
    
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [holeDistance]);
  
  // Animate fade in/out based on game state
  useFrame((_, delta) => {
    if (!groupRef.current) return;
    
    const isArmed = gameState === 'ARMED';
    const targetOpacity = isArmed ? 1 : 0;
    
    // Smooth fade
    fadeRef.current += (targetOpacity - fadeRef.current) * delta * 5;
    
    // Apply to all children
    groupRef.current.visible = fadeRef.current > 0.01;
    groupRef.current.traverse((child) => {
      if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
        const material = child.material as THREE.Material;
        if (material && 'opacity' in material) {
          material.opacity = (material.userData.baseOpacity ?? 0.3) * fadeRef.current;
        }
      }
    });
  });

  // Single dot - shows realistic tempo
  const numDots = 1;
  
  return (
    <group ref={groupRef}>
      {/* Main aim line - subtle dashed appearance */}
      <line geometry={lineGeometry}>
        <lineDashedMaterial 
          color="#8EB897"
          transparent
          opacity={0.25}
          dashSize={0.05}
          gapSize={0.03}
          userData={{ baseOpacity: 0.25 }}
        />
      </line>
      
      {/* Center line - solid but very faint */}
      <mesh position={[0, 0.002, holeDistance / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.008, holeDistance]} />
        <meshBasicMaterial 
          color="#8EB897" 
          transparent 
          opacity={0.15}
          userData={{ baseOpacity: 0.15 }}
        />
      </mesh>
      
      {/* Rolling dots - show tempo/speed indication */}
      {Array.from({ length: numDots }).map((_, i) => (
        <RollingDot 
          key={i}
          holeDistance={holeDistance}
          delay={i / numDots}
          speed={dotSpeed}
        />
      ))}
      
      {/* Target ring around hole */}
      <mesh position={[0, 0.005, holeDistance]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.08, 0.1, 32]} />
        <meshBasicMaterial 
          color="#8EB897" 
          transparent 
          opacity={0.3}
          userData={{ baseOpacity: 0.3 }}
        />
      </mesh>
      
      {/* Distance markers every meter */}
      {Array.from({ length: Math.floor(holeDistance) }).map((_, i) => (
        <mesh 
          key={`marker-${i}`} 
          position={[0, 0.002, i + 1]} 
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <planeGeometry args={[0.15, 0.003]} />
          <meshBasicMaterial 
            color="#8EB897" 
            transparent 
            opacity={0.2}
            userData={{ baseOpacity: 0.2 }}
          />
        </mesh>
      ))}
    </group>
  );
};
