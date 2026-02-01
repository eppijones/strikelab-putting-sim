import React, { useMemo, useRef } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';
import * as THREE from 'three';
import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

const StrikeLabLogo: React.FC = () => {
  const darkGreen = "#1B3B2A";
  
  return (
    <group>
      {/* Icon Group */}
      <group position={[0, 0.04, 0.001]}>
        {/* Dark Green Circle Background */}
        <mesh>
          <circleGeometry args={[0.07, 32]} />
          <meshBasicMaterial color={darkGreen} />
        </mesh>
        {/* White Crosshair Ring */}
        <mesh position={[0, 0, 0.001]}>
          <ringGeometry args={[0.02, 0.028, 32]} />
          <meshBasicMaterial color="white" />
        </mesh>
        {/* White Crosshair Ticks */}
        {/* Top */}
        <mesh position={[0, 0.04, 0.001]}>
             <planeGeometry args={[0.008, 0.02]} />
             <meshBasicMaterial color="white" />
        </mesh>
        {/* Bottom */}
        <mesh position={[0, -0.04, 0.001]}>
             <planeGeometry args={[0.008, 0.02]} />
             <meshBasicMaterial color="white" />
        </mesh>
        {/* Left */}
        <mesh position={[-0.04, 0, 0.001]}>
             <planeGeometry args={[0.02, 0.008]} />
             <meshBasicMaterial color="white" />
        </mesh>
        {/* Right */}
        <mesh position={[0.04, 0, 0.001]}>
             <planeGeometry args={[0.02, 0.008]} />
             <meshBasicMaterial color="white" />
        </mesh>
      </group>

      {/* Text */}
      <Text
        position={[0, -0.06, 0.001]}
        fontSize={0.045}
        color={darkGreen}
        anchorX="center"
        anchorY="middle"
        letterSpacing={-0.02}
      >
        StrikeLab
      </Text>
    </group>
  );
};

const AnimatedFlag: React.FC = () => {
  const groupRef = useRef<THREE.Group>(null);
  
  useFrame(({ clock }) => {
    if (groupRef.current) {
      // Gentle wind sway
      const t = clock.getElapsedTime();
      // Rotate around Y axis (flutter) - Base rotation Math.PI to point RIGHT
      groupRef.current.rotation.y = Math.PI + Math.sin(t * 2) * 0.15; // +/- 8.5 degrees around 180 deg
      // Slight lift/droop (Z axis)
      groupRef.current.rotation.z = Math.sin(t * 1.5 + 1) * 0.05; 
    }
  });

  return (
    <group ref={groupRef} position={[0, 1.875, 0]}> {/* Attached at pole top (approx) */}
        {/* The Flag Group - Offset so pivot is at the pole */}
        <group position={[0.175, 0, 0]}> {/* Center of flag is 0.175m from pole (0.35 width / 2) */}
            {/* White Flag Background */}
            <mesh>
                <planeGeometry args={[0.35, 0.25]} />
                <meshStandardMaterial 
                    color="#ffffff"
                    side={THREE.DoubleSide}
                    roughness={0.3}
                    metalness={0.1}
                />
            </mesh>
            
            {/* Front Logo - Facing Camera (+Z) */}
            {/* If flag extends +X, and camera looks -Z, then "Front" is +Z face */}
            <group position={[0, 0, 0.001]}>
                <StrikeLabLogo />
            </group>
            
            {/* Back Logo - Facing Away (-Z) */}
            <group rotation={[0, Math.PI, 0]} position={[0, 0, -0.001]}>
                <StrikeLabLogo />
            </group>
        </group>
    </group>
  );
};

// Dotted circle marker for the starting position
const StartingPositionMarker: React.FC = () => {
  const radius = 0.05; // 5cm radius - slightly larger than ball
  const dashCount = 16;
  const dashLength = 0.6; // Fraction of arc per dash (0.6 = 60% dash, 40% gap)
  
  const dashes = useMemo(() => {
    const segments = [];
    const anglePerDash = (Math.PI * 2) / dashCount;
    const dashArc = anglePerDash * dashLength;
    
    for (let i = 0; i < dashCount; i++) {
      const startAngle = i * anglePerDash;
      const endAngle = startAngle + dashArc;
      
      // Create arc segment
      const curve = new THREE.EllipseCurve(
        0, 0,           // Center
        radius, radius, // X/Y radius
        startAngle, endAngle,
        false,          // Clockwise
        0               // Rotation
      );
      
      const points = curve.getPoints(8);
      segments.push(points);
    }
    return segments;
  }, []);

  return (
    <group position={[0, 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      {dashes.map((points, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={points.length}
              array={new Float32Array(points.flatMap(p => [p.x, p.y, 0]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="rgba(255, 255, 255, 0.5)" linewidth={2} />
        </line>
      ))}
    </group>
  );
};

const GridRuler: React.FC = () => {
  const { showDistance, gameData } = usePuttingState();
  
  if (!showDistance) return null;

  const holeDistance = gameData?.hole?.distance_m ?? 3;
  const holeClearance = 0.2; // Gap around hole
  const textClearance = 0.4; // Gap around text
  const textCenter = 1.2;    // X position of text
  const greenWidth = 4;
  const halfWidth = greenWidth / 2;

  const markers = [];

  // Helper to create line segments given gaps
  const createLineSegments = (z: number, gaps: Array<[number, number]>, color: string, thickness: number) => {
    // Start with one full segment covering the green
    let segments = [[-halfWidth, halfWidth]];

    // Apply gaps sequentially
    gaps.forEach(([gapStart, gapEnd]) => {
      const newSegments: number[][] = [];
      segments.forEach(([segStart, segEnd]) => {
        // Check intersection with gap
        if (gapEnd <= segStart || gapStart >= segEnd) {
          // No overlap, keep segment
          newSegments.push([segStart, segEnd]);
        } else {
          // Overlap, split segment
          if (segStart < gapStart) {
            newSegments.push([segStart, gapStart]);
          }
          if (gapEnd < segEnd) {
            newSegments.push([gapEnd, segEnd]);
          }
        }
      });
      segments = newSegments;
    });

    return segments.map((seg, idx) => {
        const width = seg[1] - seg[0];
        const center = seg[0] + width / 2;
        return (
            <mesh key={`seg-${z}-${idx}`} position={[center, 0.002, z]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[width, thickness]} />
                <meshBasicMaterial color={color} />
            </mesh>
        );
    });
  };

  // 1. Precision Ticks (0.1m increments)
  for (let i = 1; i <= 250; i++) {
      const dist = i * 0.1;
      if (dist > 25) break;
      
      // Skip 0.5m marks (handled by main loop)
      if (Math.abs(dist % 0.5) < 0.001) continue;

      // Draw small tick on center line (if not at hole)
      if (Math.abs(dist - holeDistance) > 0.15) {
          markers.push(
            <mesh key={`tick-C-${dist}`} position={[0, 0.002, dist]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[0.1, 0.005]} /> {/* 10cm wide, thin */}
                <meshBasicMaterial color="rgba(255, 255, 255, 0.1)" />
            </mesh>
          );
      }
      
      // Ticks at the text lines (left and right) for ruler feel
      markers.push(
        <mesh key={`tick-L-${dist}`} position={[-textCenter, 0.002, dist]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[0.05, 0.005]} />
            <meshBasicMaterial color="rgba(255, 255, 255, 0.1)" />
        </mesh>
      );
      markers.push(
        <mesh key={`tick-R-${dist}`} position={[textCenter, 0.002, dist]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[0.05, 0.005]} />
            <meshBasicMaterial color="rgba(255, 255, 255, 0.1)" />
        </mesh>
      );
  }

  // 2. Main Lines (0.5m increments)
  for (let i = 1; i <= 50; i++) {
    const dist = i * 0.5;
    const isMajor = Number.isInteger(dist);
    const thickness = isMajor ? 0.015 : 0.008; // Major lines thicker
    const opacity = isMajor ? 0.2 : 0.1;
    const color = `rgba(255, 255, 255, ${opacity})`;
    
    const gaps: Array<[number, number]> = [];

    // Hole Gap
    if (Math.abs(dist - holeDistance) < 0.05) {
        gaps.push([-holeClearance/2, holeClearance/2]);
    }

    // Text Gaps (only for Major lines)
    if (isMajor) {
        gaps.push([-textCenter - textClearance/2, -textCenter + textClearance/2]);
        gaps.push([textCenter - textClearance/2, textCenter + textClearance/2]);
    }

    markers.push(...createLineSegments(dist, gaps, color, thickness));

    // Text Labels
    if (isMajor) {
        // Left Text
        markers.push(
          <Text
            key={`T-L-${dist}`}
            position={[-textCenter, 0.002, dist]}
            rotation={[-Math.PI / 2, 0, Math.PI]} // Readable from camera
            fontSize={0.15} // Slightly larger
            color="rgba(255, 255, 255, 0.4)"
            anchorX="center" // Centered in gap
            anchorY="middle"
          >
            {dist}m
          </Text>
        );

        // Right Text
        markers.push(
          <Text
            key={`T-R-${dist}`}
            position={[textCenter, 0.002, dist]}
            rotation={[-Math.PI / 2, 0, Math.PI]} // Readable from camera
            fontSize={0.15}
            color="rgba(255, 255, 255, 0.4)"
            anchorX="center"
            anchorY="middle"
          >
            {dist}m
          </Text>
        );
    }
  }

  return <group>{markers}</group>;
};

export const VirtualGreenPlane: React.FC = () => {
  const { gameData } = usePuttingState();
  
  // Get hole distance from game data, default to 3m
  const holeDistance = gameData?.hole?.distance_m ?? 3;
  const holeRadius = 0.054; // Standard golf hole radius (108mm diameter)
  
  // Clean Scandinavian Design Texture: Minimalist Green with Crisp Grid
  const gridTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      // 1. Base Color: Slightly lighter green for the putting surface to stand out
      ctx.fillStyle = '#4A6F55'; 
      ctx.fillRect(0, 0, 1024, 1024);
      
      // 2. Grid Lines: Very subtle
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';  
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
    // Texture is 1 unit. Green is 4m wide, 60m long.
    tex.repeat.set(4, 60); 
    return tex;
  }, []);

  // Create geometry with hole cutout
  const greenGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    
    // Green dimensions (matching previous PlaneGeometry args=[4, 60])
    const width = 4;
    const length = 60;
    
    // Previous position was [0, 0, 12]. PlaneGeometry is centered at local (0,0).
    // So in local coordinates (before rotation), the rect is from -width/2 to width/2 in X,
    // and -length/2 to length/2 in Y.
    // The rotation is [-Math.PI/2, 0, 0], so local Y becomes global Z.
    // Wait, if I use ShapeGeometry, the UV mapping might be different than PlaneGeometry.
    // PlaneGeometry maps 0..1 across the whole rect. ShapeGeometry might do the same if configured, 
    // or I might need to adjust texture repeat.
    // Let's stick to the local coordinates of the mesh.
    
    shape.moveTo(-width/2, -length/2);
    shape.lineTo(width/2, -length/2);
    shape.lineTo(width/2, length/2);
    shape.lineTo(-width/2, length/2);
    shape.lineTo(-width/2, -length/2);

    // Cut out the hole
    // The hole is at `holeDistance` relative to the start? 
    // In the previous code: <group position={[0, 0.01, holeDistance]}>
    // The green was at Z=12. 
    // If the ball starts at Z=0, and hole is at Z=holeDistance.
    // The green plane was at Z=12, length 35. So it spanned Z = 12 - 17.5 = -5.5 to 29.5.
    // The hole is at Z = holeDistance (e.g. 3m).
    // So in the local coordinate space of the green mesh (which is at Z=12), 
    // the hole Y-coordinate (which maps to global Z) should be `holeDistance - 12`.
    
    // Fix for hole position bug:
    // The previous calculation `holeDistance - 12` resulted in the hole appearing at 21m when distance was 3m.
    // This implies the coordinate mapping is inverted relative to the mesh center.
    // Changing to `12 - holeDistance` should align it correctly.
    const holeY = 12 - holeDistance;
    
    const holePath = new THREE.Path();
    // Use clockwise (true) for holes in ShapeGeometry to ensure they are subtracted
    holePath.absarc(0, holeY, holeRadius, 0, Math.PI * 2, true);
    shape.holes.push(holePath);

    return new THREE.ShapeGeometry(shape);
  }, [holeDistance]);

  return (
    <group>
        {/* The Green Strip with Hole Cutout */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 12]} receiveShadow geometry={greenGeometry}>
            <meshStandardMaterial 
                map={gridTexture}
                color="#ffffff"
                roughness={0.6} 
                metalness={0.05}
            />
        </mesh>
        
        {/* The Cup (Physical Depth) */}
        <group position={[0, -0.10, holeDistance]}>
             {/* Cup Walls (White Plastic) - slightly darker for depth perception */}
             <mesh position={[0, 0.05, 0]}> 
                <cylinderGeometry args={[holeRadius, holeRadius, 0.1, 32, 1, true]} /> 
                <meshStandardMaterial color="#CCCCCC" side={THREE.DoubleSide} roughness={0.3} />
             </mesh>
             {/* Cup Bottom - Darker for depth */}
             <mesh rotation={[-Math.PI/2, 0, 0]} position={[0, 0.001, 0]}>
                <circleGeometry args={[holeRadius, 32]} />
                <meshStandardMaterial color="#0a0a0a" />
             </mesh>
             {/* Inner Shadow Gradient for realistic depth */}
             <mesh position={[0, 0.05, 0]}>
                <cylinderGeometry args={[holeRadius * 0.98, holeRadius * 0.98, 0.1, 32, 1, true]} />
                <meshBasicMaterial color="#000000" opacity={0.3} transparent side={THREE.BackSide} />
             </mesh>
        </group>
        
        {/* Starting Position Marker */}
        <StartingPositionMarker />
        
        {/* Ruler Markings */}
        <GridRuler />
        
        {/* Edges - Minimalist Border */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-2.05, -0.005, 12]} receiveShadow>
            <planeGeometry args={[0.1, 60]} />
            <meshStandardMaterial color="#2F4F38" roughness={1} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[2.05, -0.005, 12]} receiveShadow>
            <planeGeometry args={[0.1, 60]} />
            <meshStandardMaterial color="#2F4F38" roughness={1} />
        </mesh>
        
        {/* Flag and Pin */}
        <group position={[0, 0, holeDistance]}>
            {/* Pin - sits in the cup */}
            <mesh position={[0, 1, 0]}>
                <cylinderGeometry args={[0.008, 0.008, 2, 8]} />
                <meshStandardMaterial color="#E0E0E0" metalness={0.5} roughness={0.2} />
            </mesh>
            
            {/* Animated Flag */}
            <AnimatedFlag />
        </group>
    </group>
  );
};
