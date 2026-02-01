import React from 'react';
import * as THREE from 'three';
import { usePuttingState } from '../../contexts/WebSocketContext';

export const VirtualGreenPlane: React.FC = () => {
  const { gameData } = usePuttingState();
  
  // Get hole distance from game data, default to 3m
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  
  // Clean Scandinavian Design Texture: Minimalist Green with Crisp Grid
  const gridTexture = React.useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      // 1. Base Color: Deep, matte forest/sage green (Scandi style)
      ctx.fillStyle = '#3A5F45'; // Deeper, more muted green
      ctx.fillRect(0, 0, 1024, 1024);
      
      // 2. Grid Lines: Very subtle
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'; 
      ctx.lineWidth = 2;
      
      const gridSize = 10; // 10x10 grid on the texture
      const step = 1024 / gridSize; 
      
      for (let i = 0; i <= 1024; i += step) {
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, 1024);
        ctx.stroke();
        
        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(1024, i);
        ctx.stroke();
      }
      
      // 3. Center Line: Slightly more visible for alignment
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(512, 0);
      ctx.lineTo(512, 1024);
      ctx.stroke();
    }
    
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    // Repeat to make 1m squares roughly
    // Texture is 1 unit. Green is 4m wide, 35m long.
    tex.repeat.set(4, 35); 
    return tex;
  }, []);

  return (
    <group>
        {/* The Green Strip - Clean and slick */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 12]} receiveShadow>
            <planeGeometry args={[4, 35]} /> {/* 4m wide, 35m long */}
            <meshStandardMaterial 
                map={gridTexture}
                color="#ffffff"
                roughness={0.6} // Reduced glare, more matte
                metalness={0.05}
            />
        </mesh>
        
        {/* Edges - Minimalist Border */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-2.05, -0.005, 12]} receiveShadow>
            <planeGeometry args={[0.1, 35]} />
            <meshStandardMaterial color="#2F4F38" roughness={1} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[2.05, -0.005, 12]} receiveShadow>
            <planeGeometry args={[0.1, 35]} />
            <meshStandardMaterial color="#2F4F38" roughness={1} />
        </mesh>
        
        {/* The Hole */}
        <group position={[0, 0.01, holeDistance]}>
            {/* Hole Cup */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.05, 0.055, 32]} />
                <meshStandardMaterial color="#FFFFFF" />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
                <circleGeometry args={[0.05, 32]} />
                <meshBasicMaterial color="#1A1A1A" />
            </mesh>
            
            {/* Flag - Simple Minimalist Design */}
            {/* Pin */}
            <mesh position={[0, 1, 0]}>
                <cylinderGeometry args={[0.008, 0.008, 2, 8]} />
                <meshStandardMaterial color="#E0E0E0" metalness={0.5} roughness={0.2} />
            </mesh>
            
            {/* Flag flag - Static, clean geometry */}
            <mesh position={[0, 1.8, 0]} rotation={[0, -Math.PI/2, 0]}>
                <boxGeometry args={[0.01, 0.25, 0.35]} />
                <meshStandardMaterial color="#D4AF37" /> {/* Muted Gold */}
            </mesh>
        </group>
    </group>
  );
};
