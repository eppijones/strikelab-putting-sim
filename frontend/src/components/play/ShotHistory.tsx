import React, { useEffect, useState } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

interface Shot {
  id?: number;
  timestamp: number;
  speed_m_s: number;
  distance_m: number;
  direction_deg: number;
  target_distance_m: number;
  result: string;
  is_made: boolean;
  distance_to_hole_m: number;
}

interface ShotHistoryProps {
  className?: string;
  onClose?: () => void;
}

export const ShotHistory: React.FC<ShotHistoryProps> = ({ className = '', onClose }) => {
  const { sessionData } = usePuttingState();
  const [history, setHistory] = useState<Shot[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedShots, setSelectedShots] = useState<Set<number>>(new Set());

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/session/history');
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setHistory(data.shots);
        }
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [sessionData?.total_putts]);

  const handleDelete = async (shotId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/shots/${shotId}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        setHistory(prev => prev.filter(s => s.id !== shotId));
        setSelectedShots(prev => {
          const newSet = new Set(prev);
          newSet.delete(shotId);
          return newSet;
        });
      }
    } catch (error) {
      console.error('Error deleting shot:', error);
    }
  };

  const deleteSelected = async () => {
    for (const id of selectedShots) {
      await handleDelete(id);
    }
    setSelectedShots(new Set());
  };

  const toggleSelect = (id: number) => {
    setSelectedShots(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) newSet.delete(id);
      else newSet.add(id);
      return newSet;
    });
  };

  return (
    <div className={`bg-black/80 backdrop-blur-md rounded-2xl border border-white/10 overflow-hidden flex flex-col ${className}`}>
      <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5">
        <h2 className="font-bold text-white flex items-center gap-2">
          <span className="uppercase tracking-wider text-xs text-white/60">SHOT HISTORY</span>
        </h2>
        {onClose && (
          <button onClick={onClose} className="text-white/40 hover:text-white">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="p-2 border-b border-white/10 flex justify-between items-center bg-black/20">
        <div className="text-xs text-white/40">
          {selectedShots.size} selected
        </div>
        {selectedShots.size > 0 && (
          <button 
            onClick={deleteSelected}
            className="px-3 py-1 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg text-xs font-bold transition-colors border border-red-500/30"
          >
            Delete Selected
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
        {loading && history.length === 0 ? (
          <div className="text-center py-8 text-white/40 text-sm">Loading...</div>
        ) : history.length === 0 ? (
          <div className="text-center py-8 text-white/40 text-sm">No shots recorded yet</div>
        ) : (
          history.map((shot, index) => (
            <div 
              key={shot.id || index}
              className={`p-3 rounded-xl border transition-all hover:bg-white/5 cursor-pointer ${
                shot.id && selectedShots.has(shot.id) 
                  ? 'border-emerald-500/50 bg-emerald-500/10' 
                  : 'border-white/5 bg-white/5'
              }`}
              onClick={() => shot.id && toggleSelect(shot.id)}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <input 
                    type="checkbox" 
                    checked={shot.id ? selectedShots.has(shot.id) : false}
                    onChange={() => {}} // Handled by parent div click
                    className="rounded border-white/20 bg-black/40 text-emerald-500 focus:ring-emerald-500/50"
                  />
                  <span className="text-xs font-mono text-white/40">
                    {new Date(shot.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                  <span className="font-bold text-white text-sm">
                    {Math.round(shot.target_distance_m * 3.28084)}ft
                  </span>
                </div>
                <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                  shot.is_made 
                    ? 'bg-emerald-500/20 text-emerald-400' 
                    : shot.result.includes('short') ? 'bg-blue-500/20 text-blue-400'
                    : shot.result.includes('long') ? 'bg-red-500/20 text-red-400'
                    : 'bg-amber-500/20 text-amber-400'
                }`}>
                  {shot.result.replace('miss_', '')}
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-white/40">Speed</span>
                  <span className="font-mono text-white/80">{shot.speed_m_s.toFixed(2)} m/s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/40">Line</span>
                  <span className="font-mono text-white/80">
                    {shot.direction_deg > 0 ? '+' : ''}{shot.direction_deg.toFixed(1)}°
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
