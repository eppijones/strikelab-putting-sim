import React, { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { usePuttingState } from '../../contexts/WebSocketContext';

const ClubPath3D: React.FC = () => {
  const { gameState, shotReport } = usePuttingState();

  const pathData = useMemo(() => {
    if (!shotReport || !shotReport.club_path_3d || shotReport.club_path_3d.length < 2) return null;

    const points: [number, number, number][] = shotReport.club_path_3d.map(
      ([x, y, z]) => [x, y, z] as [number, number, number]
    );
    return points;
  }, [shotReport]);

  const isVisible = gameState === 'STOPPED' && pathData !== null;

  if (!isVisible || !pathData) return null;

  const clubPathDeg = shotReport?.club.club_path_deg ?? 0;
  const pathColor = Math.abs(clubPathDeg) < 2 ? '#22d3ee' : clubPathDeg > 0 ? '#fb923c' : '#a78bfa';

  return (
    <group>
      {/* Club path line */}
      <Line
        points={pathData}
        color={pathColor}
        lineWidth={3}
        transparent
        opacity={0.7}
      />

      {/* Impact point indicator */}
      {pathData.length > 0 && (
        <mesh position={pathData[Math.floor(pathData.length * 0.6)]}>
          <sphereGeometry args={[0.008, 16, 16]} />
          <meshBasicMaterial color={pathColor} transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  );
};

export default ClubPath3D;
