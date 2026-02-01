import React, { useState } from 'react';
import { usePuttingState } from '../../contexts/WebSocketContext';

interface UserSelectorProps {
  className?: string;
  onClose?: () => void;
}

export const UserSelector: React.FC<UserSelectorProps> = ({ className = '', onClose }) => {
  const { users, selectUser, createUser, deleteUser, resetUserData, sessionData } = usePuttingState();
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [newHandicap, setNewHandicap] = useState('0');
  const [resetResult, setResetResult] = useState<{ userId: number; shots: number; sessions: number } | null>(null);

  const currentUserId = sessionData?.user_id;

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    
    await createUser(newName, parseFloat(newHandicap) || 0);
    setNewName('');
    setNewHandicap('0');
    setIsCreating(false);
  };

  const handleSelect = async (userId: number | null) => {
    await selectUser(userId);
    if (onClose) onClose();
  };

  return (
    <div className={`bg-black/80 backdrop-blur-md rounded-2xl border border-white/10 p-6 text-white ${className}`}>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold font-mono">Select Golfer</h2>
        {onClose && (
          <button onClick={onClose} className="text-white/50 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="space-y-2 mb-6 max-h-60 overflow-y-auto custom-scrollbar">
        <button
          onClick={() => handleSelect(null)}
          className={`w-full flex items-center justify-between p-3 rounded-xl transition-all ${
            currentUserId === null || currentUserId === undefined
              ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-100'
              : 'bg-white/5 hover:bg-white/10 border border-white/5 text-white/80'
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div className="text-left">
              <div className="font-bold">Guest</div>
              <div className="text-xs opacity-60">No stats tracking</div>
            </div>
          </div>
          {currentUserId === null && (
            <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" />
          )}
        </button>

        {users.map((user) => (
          <div key={user.id} className="group relative">
            <button
              onClick={() => handleSelect(user.id)}
              className={`w-full flex items-center justify-between p-3 rounded-xl transition-all ${
                currentUserId === user.id
                  ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-100'
                  : 'bg-white/5 hover:bg-white/10 border border-white/5 text-white/80'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-300 font-bold">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div className="text-left">
                  <div className="font-bold">{user.name}</div>
                  <div className="text-xs opacity-60">HCP: {user.handicap}</div>
                </div>
              </div>
              {currentUserId === user.id && (
                <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" />
              )}
            </button>
            
            {/* Reset data button */}
            <button
              onClick={async (e) => {
                e.stopPropagation();
                if (confirm(`Reset all data for ${user.name}? This will delete all their shots and sessions. This cannot be undone.`)) {
                  const result = await resetUserData(user.id);
                  if (result.success) {
                    setResetResult({
                      userId: user.id,
                      shots: result.shots_deleted || 0,
                      sessions: result.sessions_deleted || 0
                    });
                    // Clear the notification after 3 seconds
                    setTimeout(() => setResetResult(null), 3000);
                  }
                }
              }}
              className="absolute right-12 top-1/2 -translate-y-1/2 p-2 text-yellow-400/0 group-hover:text-yellow-400 hover:bg-yellow-500/10 rounded-lg transition-all"
              title="Reset user data"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            
            {/* Delete user button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (confirm(`Delete user ${user.name}? This will permanently delete the user and all their data. This cannot be undone.`)) {
                  deleteUser(user.id);
                }
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-red-400/0 group-hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
              title="Delete user"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        ))}
      </div>

      {/* Reset notification */}
      {resetResult && (
        <div className="mb-4 p-3 bg-emerald-500/20 border border-emerald-500/50 rounded-xl text-emerald-100 text-sm animate-pulse">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span>
              Data reset: {resetResult.shots} shots and {resetResult.sessions} sessions deleted
            </span>
          </div>
        </div>
      )}

      {isCreating ? (
        <form onSubmit={handleCreate} className="space-y-3 bg-white/5 p-4 rounded-xl border border-white/10">
          <div>
            <label className="block text-xs text-white/60 mb-1">Name</label>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500/50 text-white"
              placeholder="Enter name"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-xs text-white/60 mb-1">Handicap</label>
            <input
              type="number"
              step="0.1"
              value={newHandicap}
              onChange={(e) => setNewHandicap(e.target.value)}
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500/50 text-white"
            />
          </div>
          <div className="flex gap-2 pt-2">
            <button
              type="button"
              onClick={() => setIsCreating(false)}
              className="flex-1 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-xs font-bold transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-xs font-bold transition-colors shadow-lg shadow-emerald-900/20"
            >
              Create User
            </button>
          </div>
        </form>
      ) : (
        <button
          onClick={() => setIsCreating(true)}
          className="w-full py-3 border border-dashed border-white/20 hover:border-white/40 rounded-xl text-white/60 hover:text-white transition-all flex items-center justify-center gap-2 group"
        >
          <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center group-hover:bg-emerald-500/20 group-hover:text-emerald-400 transition-colors">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </div>
          <span className="font-mono text-sm">Add New Golfer</span>
        </button>
      )}
    </div>
  );
};
