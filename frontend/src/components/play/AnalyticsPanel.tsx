import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart3, TrendingUp, Target, X, RefreshCw } from 'lucide-react';

interface StatsData {
  total_putts: number;
  putts_made: number;
  make_percentage: number;
  avg_speed_m_s: number;
  avg_distance_m: number;
  avg_line_error_deg: number;
  by_distance: Record<string, { total: number; made: number; percentage: number }>;
  by_result: Record<string, { count: number; percentage: number }>;
}

interface TrendData {
  timestamps: string[];
  speeds: number[];
  distances: number[];
  directions: number[];
  results: string[];
  made: boolean[];
  distance_to_hole: number[];
}

interface AnalyticsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

// Simple bar chart component
const BarChart: React.FC<{
  data: { label: string; value: number; maxValue?: number }[];
  height?: number;
  barColor?: string;
}> = ({ data, height = 120, barColor = 'rgb(52, 211, 153)' }) => {
  const maxValue = Math.max(...data.map(d => d.maxValue || d.value), 1);
  
  return (
    <div className="flex items-end gap-2 justify-around" style={{ height }}>
      {data.map((item, i) => {
        const barHeight = (item.value / maxValue) * (height - 30);
        return (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className="text-xs text-white/70 font-medium">
              {item.value.toFixed(0)}%
            </div>
            <div
              className="w-8 rounded-t transition-all duration-300"
              style={{
                height: barHeight,
                backgroundColor: barColor,
                opacity: item.value > 0 ? 1 : 0.3
              }}
            />
            <div className="text-[10px] text-white/50 text-center">
              {item.label}
            </div>
          </div>
        );
      })}
    </div>
  );
};

// Dispersion chart - shows where shots end up
const DispersionChart: React.FC<{
  data: { lateral: number; depth: number; made: boolean }[];
}> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Vertical line
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();
    
    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    
    // Draw hole (center)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw tolerance circle
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.3)';
    ctx.beginPath();
    ctx.arc(centerX, centerY, 20, 0, Math.PI * 2);
    ctx.stroke();
    
    // Scale factor (pixels per meter)
    const scale = 80;
    
    // Draw shots
    data.forEach(shot => {
      const x = centerX + shot.lateral * scale;
      const y = centerY - shot.depth * scale; // Negative because Y is inverted
      
      ctx.fillStyle = shot.made ? 'rgb(52, 211, 153)' : 'rgb(239, 68, 68)';
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
    
    ctx.globalAlpha = 1;
    
    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SHORT', centerX, height - 5);
    ctx.fillText('LONG', centerX, 12);
    ctx.textAlign = 'left';
    ctx.fillText('LEFT', 5, centerY - 5);
    ctx.textAlign = 'right';
    ctx.fillText('RIGHT', width - 5, centerY - 5);
    
  }, [data]);
  
  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={200}
      className="bg-slate-900/50 rounded-lg"
    />
  );
};

export const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({ isOpen, onClose }) => {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [trend, setTrend] = useState<TrendData | null>(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState<'session' | 'week' | 'all'>('all');
  
  const fetchStats = async () => {
    setLoading(true);
    try {
      const endpoint = period === 'week' 
        ? '/api/stats/recent' 
        : '/api/stats/all-time';
      
      const [statsRes, trendRes] = await Promise.all([
        fetch(`http://localhost:8000${endpoint}`),
        fetch('http://localhost:8000/api/stats/trend')
      ]);
      
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
      
      if (trendRes.ok) {
        const trendData = await trendRes.json();
        setTrend(trendData);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
    setLoading(false);
  };
  
  useEffect(() => {
    if (isOpen) {
      fetchStats();
    }
  }, [isOpen, period]);
  
  // Build dispersion data from trend
  const dispersionData = trend ? trend.distance_to_hole.map((dist, i) => ({
    lateral: trend.directions[i] ? Math.sin(trend.directions[i] * Math.PI / 180) * dist : 0,
    depth: trend.distance_to_hole[i] || 0,
    made: trend.made[i]
  })).slice(-30) : [];
  
  // Build distance chart data
  const distanceChartData = stats?.by_distance 
    ? Object.entries(stats.by_distance).map(([label, data]) => ({
        label: label.replace('m', ''),
        value: data.percentage || 0,
        maxValue: 100
      }))
    : [];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
            onClick={onClose}
          />
          
          {/* Panel */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-4 md:inset-10 bg-slate-900/95 backdrop-blur-xl 
                     rounded-3xl border border-white/10 z-50 overflow-hidden flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div className="flex items-center gap-3">
                <BarChart3 size={24} className="text-emerald-400" />
                <h2 className="text-xl font-bold text-white">Analytics</h2>
              </div>
              <div className="flex items-center gap-4">
                {/* Period selector */}
                <div className="flex gap-1 bg-white/5 rounded-lg p-1">
                  {(['all', 'week'] as const).map(p => (
                    <button
                      key={p}
                      onClick={() => setPeriod(p)}
                      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                                ${period === p 
                                  ? 'bg-emerald-500 text-white' 
                                  : 'text-white/60 hover:text-white'}`}
                    >
                      {p === 'all' ? 'All Time' : 'Last 7 Days'}
                    </button>
                  ))}
                </div>
                
                {/* Refresh */}
                <button
                  onClick={fetchStats}
                  disabled={loading}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <RefreshCw size={18} className={`text-white/60 ${loading ? 'animate-spin' : ''}`} />
                </button>
                
                {/* Close */}
                <button
                  onClick={onClose}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <X size={20} className="text-white/60" />
                </button>
              </div>
            </div>
            
            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {loading && !stats ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-white/50">Loading...</div>
                </div>
              ) : !stats ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-white/50">No data available yet. Make some putts!</div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {/* Overall Stats */}
                  <div className="bg-slate-800/50 rounded-2xl p-6 border border-white/5">
                    <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
                      Overview
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-white/70">Total Putts</span>
                        <span className="text-2xl font-bold text-white">{stats.total_putts}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-white/70">Made</span>
                        <span className="text-2xl font-bold text-emerald-400">{stats.putts_made}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-white/70">Make %</span>
                        <span className="text-2xl font-bold text-white">{stats.make_percentage}%</span>
                      </div>
                      <div className="h-px bg-white/10 my-2" />
                      <div className="flex justify-between items-center">
                        <span className="text-white/70">Avg Speed</span>
                        <span className="text-lg font-medium text-white">{stats.avg_speed_m_s} m/s</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-white/70">Avg Line Error</span>
                        <span className="text-lg font-medium text-white">{stats.avg_line_error_deg}°</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Make % by Distance */}
                  <div className="bg-slate-800/50 rounded-2xl p-6 border border-white/5">
                    <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
                      Make % by Distance
                    </h3>
                    {distanceChartData.length > 0 ? (
                      <BarChart data={distanceChartData} />
                    ) : (
                      <div className="h-32 flex items-center justify-center text-white/30">
                        No data
                      </div>
                    )}
                  </div>
                  
                  {/* Miss Pattern */}
                  <div className="bg-slate-800/50 rounded-2xl p-6 border border-white/5">
                    <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
                      Miss Pattern
                    </h3>
                    <div className="flex justify-center">
                      <DispersionChart data={dispersionData} />
                    </div>
                  </div>
                  
                  {/* Result Breakdown */}
                  <div className="bg-slate-800/50 rounded-2xl p-6 border border-white/5">
                    <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
                      Result Breakdown
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(stats.by_result || {}).map(([result, data]) => (
                        <div key={result} className="flex items-center gap-3">
                          <div className="flex-1">
                            <div className="flex justify-between mb-1">
                              <span className="text-sm text-white/70 capitalize">
                                {result.replace('_', ' ')}
                              </span>
                              <span className="text-sm text-white/50">
                                {data.count} ({data.percentage}%)
                              </span>
                            </div>
                            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all ${
                                  result === 'made' ? 'bg-emerald-400' :
                                  result === 'lip_out' ? 'bg-yellow-400' :
                                  'bg-red-400/60'
                                }`}
                                style={{ width: `${data.percentage}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Recent Trend (if we have trend data) */}
                  {trend && trend.made.length > 0 && (
                    <div className="bg-slate-800/50 rounded-2xl p-6 border border-white/5 md:col-span-2">
                      <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
                        Recent Shots
                      </h3>
                      <div className="flex gap-1 flex-wrap">
                        {trend.made.slice(-50).map((made, i) => (
                          <div
                            key={i}
                            className={`w-3 h-3 rounded-sm ${made ? 'bg-emerald-400' : 'bg-red-400/60'}`}
                            title={`Shot ${i + 1}: ${made ? 'Made' : 'Miss'}`}
                          />
                        ))}
                      </div>
                      <div className="flex gap-4 mt-3 text-xs text-white/40">
                        <span className="flex items-center gap-1">
                          <div className="w-2 h-2 bg-emerald-400 rounded-sm" /> Made
                        </span>
                        <span className="flex items-center gap-1">
                          <div className="w-2 h-2 bg-red-400/60 rounded-sm" /> Miss
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
