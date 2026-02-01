import React from 'react';
import * as THREE from 'three';

export const VirtualGreenPlane: React.FC = () => {
  // Procedural texture generation or standard material for now
  // A dark, premium felt/green look
  
  // Grid lines
  const gridTexture = React.useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#1E3A2B'; // Base forest
      ctx.fillRect(0, 0, 1024, 1024);
      
      // Grid
      ctx.strokeStyle = 'rgba(142, 184, 151, 0.15)'; // Sage, faint
      ctx.lineWidth = 2;
      const step = 1024 / 10; // 10 lines
      
      for (let i = 0; i <= 1024; i += step) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, 1024);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(1024, i);
        ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(5, 50); // High repeat for long green
    return tex;
  }, []);

  return (
    <group>
        {/* The Green Strip */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 10]} receiveShadow>
            <planeGeometry args={[2, 30]} /> {/* 2m wide, 30m long */}
            <meshStandardMaterial 
                map={gridTexture}
                color="#ffffff"
                roughness={0.8}
                metalness={0.1}
            />
        </mesh>
        
        {/* The Hole (at 3m usually) */}
        <group position={[0, 0.01, 3]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.054, 0.06, 32]} /> {/* Lip */}
                <meshBasicMaterial color="#8EB897" />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
                <circleGeometry args={[0.054, 32]} /> {/* Hole dark */}
                <meshBasicMaterial color="#000000" />
            </mesh>
            {/* Pin */}
            <mesh position={[0, 1, 0]}>
                <cylinderGeometry args={[0.01, 0.01, 2]} />
                <meshStandardMaterial color="#D4C5A8" metalness={0.8} roughness={0.2} />
            </mesh>
            <mesh position={[0, 1.8, 0]} rotation={[0, -Math.PI/2, 0]}>
                <boxGeometry args={[0.01, 0.3, 0.4]} />
                <meshStandardMaterial color="#F59E0B" />
            </mesh>
        </group>
    </group>
  );
};
