import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';

export const Ball3D: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { ballPosition, pixelsPerMeter, gameState, lastJsonMessage, gameData } = usePuttingState();
  
  // Physics state
  const [isDropped, setIsDropped] = useState(false);
  const dropTimeRef = useRef<number | null>(null);
  
  // Interaction state (Ref to avoid re-renders during frame loop)
  const isInteractingRef = useRef(false);
  const localVelocityRef = useRef(new THREE.Vector3());
  const localPositionRef = useRef(new THREE.Vector3());
  
  // Origin offset - captures the starting X and Z position to normalize ball to center
  // This ensures ball always appears at the starting circle regardless of physical placement
  const startOffsetXRef = useRef<number | null>(null);
  const startOffsetZRef = useRef<number | null>(null);
  const prevGameStateRef = useRef<string | null>(null);
  
  // Frozen position - captures final ball position when stopping to prevent snap-back
  const frozenPositionRef = useRef<THREE.Vector3 | null>(null);
  
  // Reset state when arming
  useEffect(() => {
    if (gameState === 'ARMED') {
      setIsDropped(false);
      dropTimeRef.current = null;
      isInteractingRef.current = false;
      localVelocityRef.current.set(0, 0, 0);
      localPositionRef.current.set(0, 0, 0);
      startOffsetXRef.current = null; // Reset X origin offset
      startOffsetZRef.current = null; // Reset Z origin offset
      frozenPositionRef.current = null; // Reset frozen position
    }
  }, [gameState]);

  useFrame((state, delta) => {
    if (!meshRef.current) return;

    const holeDistance = gameData?.hole?.distance_m ?? 3;
    const holeRadius = 0.054; // 5.4cm radius (standard golf hole)
    const dropDepth = -0.1; // Bottom of cup
    const ballRadius = 0.0213;

    
    // Capture frozen position when transitioning to STOPPED (unless ball is dropped in hole)
    if (gameState === 'STOPPED' && !frozenPositionRef.current && !isDropped) {
      // Freeze the ball at its current visual position
      frozenPositionRef.current = meshRef.current.position.clone();
    }
    
    // If we have a frozen position and we're STOPPED, use it and skip further calculations
    if (gameState === 'STOPPED' && frozenPositionRef.current && !isDropped) {
      meshRef.current.position.copy(frozenPositionRef.current);
      prevGameStateRef.current = gameState;
      return;
    }

    // 1. Get Tracker Position (Raw)
    let trackerPos = new THREE.Vector3(0, ballRadius, 0);
    let trackerSpeed = 0;

    // Get the final predicted distance from shot data (this is the authoritative stopping point)
    const shotFinalDistance = lastJsonMessage?.shot?.distance_m;

    if (ballPosition) {
      const rawZMeters = ballPosition.x / pixelsPerMeter;
      
      const resolution = lastJsonMessage?.resolution || [1280, 800];
      const centerY = resolution[1] / 2;
      const rawXMeters = (centerY - ballPosition.y) / pixelsPerMeter;
      
      // ARMED state: Continuously update offset so ball ALWAYS appears at center
      // This allows repositioning the ball before the shot
      // TRACKING/VIRTUAL_ROLLING: Lock the offset so movement is tracked relative to start
      if (gameState === 'ARMED') {
        // Always keep ball centered while setting up
        startOffsetXRef.current = rawXMeters;
        startOffsetZRef.current = rawZMeters;
      } else if (startOffsetXRef.current === null || startOffsetZRef.current === null) {
        // First frame of tracking - capture the offset if not already set
        startOffsetXRef.current = rawXMeters;
        startOffsetZRef.current = rawZMeters;
      }
      
      // Normalize both X and Z positions relative to starting position
      // Ball always starts at center (0, 0) and movement is relative to that
      const normalizedX = rawXMeters - (startOffsetXRef.current ?? 0);
      const normalizedZ = rawZMeters - (startOffsetZRef.current ?? 0);
      
      // NOTE: Don't clamp Z position before hole interaction check!
      // We need to let the ball pass through the hole area first
      const clampedX = Math.max(-2, Math.min(2, normalizedX));
      
      // Use unclamped Z for tracker position during rolling
      // This ensures the ball can pass through the hole area
      trackerPos.set(clampedX, ballRadius, normalizedZ);
      
      if (lastJsonMessage?.velocity?.speed_px_s) {
        trackerSpeed = lastJsonMessage.velocity.speed_px_s / pixelsPerMeter;
      } else if (lastJsonMessage?.virtual_ball?.speed_m_s) {
        trackerSpeed = lastJsonMessage.virtual_ball.speed_m_s;
      }
    } else if (gameState === 'ARMED') {
       // Reset
       meshRef.current.position.lerp(new THREE.Vector3(0, ballRadius, 0), delta * 5);
       meshRef.current.rotation.set(0, 0, 0);
       prevGameStateRef.current = gameState;
       return;
    }
    
    prevGameStateRef.current = gameState;

    // 2. Physics Interaction Logic
    
    // Calculate distance to hole center
    const dx = trackerPos.x - 0;
    const dz = trackerPos.z - holeDistance;
    const distToHoleCenter = Math.sqrt(dx*dx + dz*dz);
    
    // Interaction Zone: 4x radius (approx 21.6cm) - increased for better detection
    const interactionZone = holeRadius * 4.0;

    // Enter Interaction Mode if near hole (lowered speed threshold for better detection)
    // Also trigger if ball is past the hole but close laterally (passed through hole area)
    const ballPassedHole = trackerPos.z >= holeDistance;
    const laterallyAligned = Math.abs(dx) < holeRadius * 2; // Within 2x hole radius laterally
    const nearHoleOrPassedThrough = distToHoleCenter < interactionZone || 
                                     (ballPassedHole && laterallyAligned && Math.abs(dz) < 0.3);
    
    if (!isInteractingRef.current && nearHoleOrPassedThrough && trackerSpeed > 0.01) {
       isInteractingRef.current = true;
       // Initialize local physics from tracker
       localPositionRef.current.copy(trackerPos);
       
       // Set velocity - use tracker speed in the direction of travel
       // Calculate direction from ball's approach angle
       const dirZ = trackerPos.z > 0.1 ? 1 : 0; // Forward direction
       localVelocityRef.current.set(0, 0, trackerSpeed * dirZ); 
    }

    let finalPos = trackerPos;

    if (isInteractingRef.current) {
        // --- PHYSICS SIMULATION STEP ---
        
        // 1. Update position based on velocity
        const moveStep = localVelocityRef.current.clone().multiplyScalar(delta);
        localPositionRef.current.add(moveStep);
        
        // 2. Calculate forces
        const toHoleX = 0 - localPositionRef.current.x;
        const toHoleZ = holeDistance - localPositionRef.current.z;
        const distToCenter = Math.sqrt(toHoleX*toHoleX + toHoleZ*toHoleZ);
        
        // Gravity / Slope Effect toward hole
        // If near the hole, gravity pulls it in (simulates cup edge slope)
        if (distToCenter < holeRadius * 2.5) {
            // Stronger pull closer to the hole
            const pullFactor = 1 - (distToCenter / (holeRadius * 2.5));
            const pullStrength = 8.0 * pullFactor * delta;
            
            // Add force towards center (normalized)
            if (distToCenter > 0.001) {
                localVelocityRef.current.x += (toHoleX / distToCenter) * pullStrength;
                localVelocityRef.current.z += (toHoleZ / distToCenter) * pullStrength;
            }
        }
        
        // 3. Collision / Drop Logic
        if (distToCenter < holeRadius) {
            // INSIDE HOLE
            const speed = localVelocityRef.current.length();
            const MAX_DROP_SPEED = 2.0; // Increased from 1.5 - more forgiving
            
            if (speed < MAX_DROP_SPEED || isDropped) {
                // DROP INTO HOLE
                if (!isDropped) {
                    setIsDropped(true);
                    dropTimeRef.current = state.clock.elapsedTime;
                    console.log('[Ball3D] Ball dropped into hole! Speed:', speed.toFixed(2), 'm/s');
                    // Kill lateral velocity to suck it in
                    localVelocityRef.current.multiplyScalar(0.3);
                }
                
                if (dropTimeRef.current) {
                    const timeInHole = state.clock.elapsedTime - dropTimeRef.current;
                    const dropProgress = Math.min(timeInHole * 3, 1); // Slower drop for visual effect
                    
                    // Fall down into cup
                    localPositionRef.current.y = THREE.MathUtils.lerp(ballRadius, dropDepth, dropProgress);
                    
                    // Center X/Z toward hole center
                    localPositionRef.current.x = THREE.MathUtils.lerp(localPositionRef.current.x, 0, dropProgress * 0.3);
                    localPositionRef.current.z = THREE.MathUtils.lerp(localPositionRef.current.z, holeDistance, dropProgress * 0.3);
                    
                    // Dampen velocity
                    localVelocityRef.current.multiplyScalar(0.85);
                }
            } else {
                // TOO FAST - FLY OVER / LIP OUT
                console.log('[Ball3D] Ball too fast for hole! Speed:', speed.toFixed(2), 'm/s');
            }
        } else if (distToCenter < holeRadius * 1.2) {
             // LIP INTERACTION (Rim) - ball is grazing the edge
             const speed = localVelocityRef.current.length();
             
             // If slow enough and moving toward center, pull it in
             const dot = localVelocityRef.current.x * toHoleX + localVelocityRef.current.z * toHoleZ;
             if (dot > 0 && speed < 0.8) {
                 // Moving toward center and slow - increase pull
                 const pullStrength = 3.0 * delta;
                 localVelocityRef.current.x += (toHoleX / distToCenter) * pullStrength;
                 localVelocityRef.current.z += (toHoleZ / distToCenter) * pullStrength;
             }
        }

        // Friction - realistic putting green deceleration
        const frictionRate = 0.8; // Higher = more friction
        localVelocityRef.current.multiplyScalar(1 - (frictionRate * delta));
        
        // Stop if very slow
        if (localVelocityRef.current.length() < 0.02 && !isDropped) {
            localVelocityRef.current.set(0, 0, 0);
        }
        
        finalPos = localPositionRef.current.clone();
        
    } else {
        // Not interacting - follow tracker
        // Apply Z clamping only when NOT interacting with hole area
        // This ensures ball stops at predicted distance if it misses the hole
        if (shotFinalDistance && (gameState === 'VIRTUAL_ROLLING' || gameState === 'STOPPED')) {
            trackerPos.z = Math.min(trackerPos.z, shotFinalDistance);
        }
        finalPos = trackerPos;
    }

    // Apply to mesh
    // If interacting, use direct position (physics authoritative)
    // If tracking, lerp for smoothness
    if (isInteractingRef.current) {
        meshRef.current.position.copy(finalPos);
    } else {
        meshRef.current.position.lerp(finalPos, delta * 25);
    }

    // Rotation (ball rolling animation)
    const speed = isInteractingRef.current ? localVelocityRef.current.length() : trackerSpeed;
    if (speed > 0.05) {
        // Rotate around X axis proportional to speed
        meshRef.current.rotation.x += speed * delta * 5; 
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
