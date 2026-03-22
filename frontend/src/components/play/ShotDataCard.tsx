import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePuttingState, type ShotReportData } from '../../contexts/WebSocketContext';
import {
  Target, Gauge, ArrowRight, RotateCcw, Timer, TrendingUp,
  Crosshair, ArrowUpRight, Mountain, ChevronDown
} from 'lucide-react';

interface MetricTileProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit: string;
  highlight?: boolean;
  small?: boolean;
}

const MetricTile: React.FC<MetricTileProps> = ({ icon, label, value, unit, highlight, small }) => (
  <div className={`
    flex flex-col gap-1 rounded-xl
    ${highlight ? 'bg-white/10 border border-white/15' : 'bg-white/5 border border-white/8'}
    ${small ? 'p-2' : 'p-3'}
  `}>
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-white/50 font-medium">
      {icon}
      {label}
    </div>
    <div className="flex items-baseline gap-1">
      <span className={`font-bold text-white ${small ? 'text-lg' : 'text-2xl'}`}>
        {value}
      </span>
      <span className="text-xs text-white/50 font-medium">{unit}</span>
    </div>
  </div>
);

const ShotDataCard: React.FC = () => {
  const { gameState, shotReport } = usePuttingState();
  const [visible, setVisible] = useState(false);
  const [displayReport, setDisplayReport] = useState<ShotReportData | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (gameState === 'STOPPED' && shotReport) {
      setDisplayReport(shotReport);
      setVisible(true);
      setExpanded(false);
    }
    if (gameState === 'ARMED') {
      const timer = setTimeout(() => {
        setVisible(false);
        setDisplayReport(null);
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [gameState, shotReport]);

  if (!displayReport) return null;

  const { ball, club } = displayReport;
  const isChip = displayReport.shot_type === 'chip';
  const hasClub = club?.available;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, x: -40, scale: 0.95 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: -40, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed bottom-24 left-6 z-[60] w-80 pointer-events-auto"
        >
          <div className="
            bg-[#1a1f25] border border-white/15
            rounded-2xl shadow-2xl overflow-hidden
          ">
            {/* Header */}
            <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isChip ? 'bg-orange-400' : 'bg-emerald-400'}`} />
                <span className="text-sm font-semibold text-white uppercase tracking-wide">
                  {isChip ? 'Chip' : 'Putt'} Analysis
                </span>
              </div>
              <div className="flex items-center gap-1">
                {displayReport.cameras_used.map(cam => (
                  <span key={cam} className="text-[9px] px-1.5 py-0.5 rounded bg-white/10 text-white/50 uppercase font-medium">
                    {cam}
                  </span>
                ))}
              </div>
            </div>

            {/* Primary Metrics */}
            <div className="p-3 grid grid-cols-3 gap-2">
              <MetricTile
                icon={<Gauge size={10} />}
                label="Speed"
                value={ball.speed_m_s.toFixed(1)}
                unit="m/s"
                highlight
              />
              <MetricTile
                icon={<Target size={10} />}
                label="Distance"
                value={ball.distance_m.toFixed(2)}
                unit="m"
                highlight
              />
              <MetricTile
                icon={<ArrowRight size={10} />}
                label="Line"
                value={Math.abs(ball.direction_deg).toFixed(1)}
                unit={`° ${ball.direction_deg > 0 ? 'R' : 'L'}`}
                highlight
              />
            </div>

            {/* Club Metrics (if ZED available) */}
            {hasClub && (
              <div className="px-3 pb-2 grid grid-cols-3 gap-2">
                <MetricTile
                  icon={<Crosshair size={10} />}
                  label="Club Path"
                  value={Math.abs(club.club_path_deg).toFixed(1)}
                  unit={`° ${club.club_path_deg > 0 ? 'I→O' : 'O→I'}`}
                  small
                />
                <MetricTile
                  icon={<ArrowRight size={10} />}
                  label="Face"
                  value={Math.abs(club.face_angle_deg).toFixed(1)}
                  unit={`° ${club.face_angle_deg > 0 ? 'open' : 'closed'}`}
                  small
                />
                <MetricTile
                  icon={<Timer size={10} />}
                  label="Tempo"
                  value={club.stroke_tempo.toFixed(1)}
                  unit="ratio"
                  small
                />
              </div>
            )}

            {/* Chipping Metrics */}
            {isChip && (
              <div className="px-3 pb-2 grid grid-cols-3 gap-2">
                <MetricTile
                  icon={<ArrowUpRight size={10} />}
                  label="Launch"
                  value={ball.launch_angle_deg.toFixed(1)}
                  unit="°"
                  small
                />
                <MetricTile
                  icon={<Mountain size={10} />}
                  label="Peak"
                  value={(ball.peak_height_m * 100).toFixed(0)}
                  unit="cm"
                  small
                />
                <MetricTile
                  icon={<TrendingUp size={10} />}
                  label="Carry"
                  value={ball.carry_distance_m.toFixed(2)}
                  unit="m"
                  small
                />
              </div>
            )}

            {/* Expandable Section */}
            <button
              onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
              className="w-full px-4 py-2.5 flex items-center justify-center gap-1.5 text-xs font-medium text-white/60 hover:text-white hover:bg-white/5 active:bg-white/10 transition-colors border-t border-white/10 cursor-pointer select-none"
            >
              {expanded ? 'Less' : 'More Details'}
              <ChevronDown size={12} className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {expanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="px-3 pb-3 grid grid-cols-2 gap-2">
                    {!isChip && (
                      <>
                        <MetricTile
                          icon={<RotateCcw size={10} />}
                          label="True Roll"
                          value={ball.true_roll_pct.toFixed(0)}
                          unit="%"
                          small
                        />
                        <MetricTile
                          icon={<ArrowRight size={10} />}
                          label="Skid"
                          value={(ball.skid_distance_m * 100).toFixed(0)}
                          unit="cm"
                          small
                        />
                      </>
                    )}
                    {isChip && (
                      <>
                        <MetricTile
                          icon={<RotateCcw size={10} />}
                          label="Spin Est."
                          value={ball.spin_estimate_rpm.toFixed(0)}
                          unit="rpm"
                          small
                        />
                        <MetricTile
                          icon={<ChevronDown size={10} />}
                          label="Landing"
                          value={ball.landing_angle_deg.toFixed(0)}
                          unit="°"
                          small
                        />
                      </>
                    )}
                    {hasClub && (
                      <>
                        <MetricTile
                          icon={<Gauge size={10} />}
                          label="Club Speed"
                          value={club.club_speed_m_s.toFixed(1)}
                          unit="m/s"
                          small
                        />
                        <MetricTile
                          icon={<ArrowUpRight size={10} />}
                          label="Attack"
                          value={Math.abs(club.attack_angle_deg).toFixed(1)}
                          unit={`° ${club.attack_angle_deg < 0 ? 'down' : 'up'}`}
                          small
                        />
                      </>
                    )}
                  </div>
                  {displayReport.fast_putt_resolved && (
                    <div className="px-4 pb-2 text-[9px] text-emerald-400/60">
                      Speed resolved from depth cameras (fast putt)
                    </div>
                  )}
                  {displayReport.fast_putt_estimated && (
                    <div className="px-4 pb-2 text-[9px] text-amber-400/60">
                      Speed estimated — depth cameras unavailable (fast putt)
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ShotDataCard;
