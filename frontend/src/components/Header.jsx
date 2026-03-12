import React from 'react';
import { Activity } from 'lucide-react';

export default function Header({ connected }) {
  return (
    <header className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-green/10">
          <Activity className="h-5 w-5 text-accent-green" />
        </div>
        <div>
          <h1 className="font-display text-lg font-semibold tracking-tight">MacroPulse</h1>
          <p className="text-xs text-white/40">Macro Regime Intelligence</p>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xs">
          <span
            className="pulse-dot"
            style={{ color: connected ? '#00e5a0' : '#ff4d6a', background: connected ? '#00e5a0' : '#ff4d6a' }}
          />
          <span className="text-white/50">{connected ? 'Live' : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
