import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts';
import { REGIME_CONFIG, formatDate } from '../lib/utils';

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const cfg = REGIME_CONFIG[d.regime] || {};
  return (
    <div className="rounded-xl border border-white/10 bg-surface-2 px-4 py-3 shadow-xl text-sm">
      <p className="font-mono text-xs text-white/40">{formatDate(d.timestamp)}</p>
      <p className="mt-1 font-semibold" style={{ color: cfg.color }}>{cfg.label}</p>
      <p className="text-xs text-white/50">Risk: {d.risk_score}</p>
    </div>
  );
}

export default function RegimeTimeline({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="glass-card flex h-72 items-center justify-center">
        <p className="text-sm text-white/30">No history data</p>
      </div>
    );
  }

  const data = [...history]
    .reverse()
    .map((r) => ({
      timestamp: r.timestamp,
      risk_score: r.risk_score,
      regime: r.macro_regime,
    }));

  return (
    <div className="glass-card p-6 animate-fade-up" style={{ animationDelay: '0.1s' }}>
      <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-white/40">
        Risk Score Timeline
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="timestamp"
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
            tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[-100, 100]}
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="risk_score"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#riskGrad)"
            dot={false}
            activeDot={{ r: 4, fill: '#3b82f6' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
