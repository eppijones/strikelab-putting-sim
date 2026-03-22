import React, { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { usePuttingState } from '../../contexts/WebSocketContext';

const BallFlight3D: React.FC = () => {
  const { gameState, shotReport } = usePuttingState();

  const flightPath = useMemo(() => {
    if (!shotReport || shotReport.shot_type !== 'chip') return null;
    if (!shotReport.trajectory_3d || shotReport.trajectory_3d.length < 2) return null;

    const points: [number, number, number][] = shotReport.trajectory_3d.map(
      ([x, y, z]) => [x, y, z] as [number, number, number]
    );
    return points;
  }, [shotReport]);

  const isVisible = gameState === 'STOPPED' && flightPath !== null;

  if (!isVisible || !flightPath) return null;

  return (
    <group>
      {/* Flight arc */}
      <Line
        points={flightPath}
        color="#f59e0b"
        lineWidth={2}
        dashed
        dashScale={20}
        dashSize={0.05}
        gapSize={0.02}
      />

      {/* Landing marker */}
      {flightPath.length > 0 && (
        <mesh position={flightPath[flightPath.length - 1]} rotation-x={-Math.PI / 2}>
          <ringGeometry args={[0.02, 0.03, 32]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.6} />
        </mesh>
      )}

      {/* Peak height marker */}
      {shotReport && shotReport.ball.peak_height_m > 0.02 && (
        <mesh position={[0, shotReport.ball.peak_height_m, -shotReport.ball.carry_distance_m / 2]}>
          <sphereGeometry args={[0.005, 16, 16]} />
          <meshBasicMaterial color="#f59e0b" transparent opacity={0.4} />
        </mesh>
      )}
    </group>
  );
};

export default BallFlight3D;
