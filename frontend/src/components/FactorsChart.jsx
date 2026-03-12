import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import { formatDate } from '../lib/utils';

const FACTOR_COLORS = ['#00e5a0', '#3b82f6', '#ffb800', '#ff4d6a'];

function FactorTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="rounded-xl border border-white/10 bg-surface-2 px-4 py-3 shadow-xl text-sm">
      <p className="font-mono text-xs text-white/40">{formatDate(d?.time)}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="text-xs">
          {p.name}: <span className="font-mono font-semibold">{p.value?.toFixed(3)}</span>
        </p>
      ))}
    </div>
  );
}

export default function FactorsChart({ data }) {
  if (!data?.data?.length) {
    return (
      <div className="glass-card flex h-64 items-center justify-center">
        <p className="text-sm text-white/30">No factor data</p>
      </div>
    );
  }

  const rows = [...data.data].reverse();

  return (
    <div className="glass-card p-6 animate-fade-up" style={{ animationDelay: '0.3s' }}>
      <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-white/40">
        PCA Latent Factors
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="time"
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
            tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            axisLine={false}
            tickLine={false}
          />
          <YAxis tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }} axisLine={false} tickLine={false} />
          <Tooltip content={<FactorTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }}
          />
          {[1, 2, 3, 4].map((n, i) => (
            <Line
              key={n}
              type="monotone"
              dataKey={`factor_${n}`}
              name={`F${n}`}
              stroke={FACTOR_COLORS[i]}
              strokeWidth={1.5}
              dot={false}
              opacity={0.8}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
