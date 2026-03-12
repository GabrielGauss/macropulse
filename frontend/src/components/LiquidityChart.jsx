import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { formatDate, formatNumber } from '../lib/utils';

function LiqTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-xl border border-white/10 bg-surface-2 px-4 py-3 shadow-xl text-sm">
      <p className="font-mono text-xs text-white/40">{formatDate(d.time)}</p>
      <p className="mt-1 text-white/80">Net Liq: <span className="font-mono font-semibold text-accent-purple">{formatNumber(d.net_liquidity)}</span></p>
      <p className="text-xs text-white/40">Δ: {d.d_liquidity != null ? formatNumber(d.d_liquidity) : '—'}</p>
    </div>
  );
}

export default function LiquidityChart({ data }) {
  if (!data?.data?.length) {
    return (
      <div className="glass-card flex h-64 items-center justify-center">
        <p className="text-sm text-white/30">No liquidity data</p>
      </div>
    );
  }

  const rows = [...data.data].reverse();

  return (
    <div className="glass-card p-6 animate-fade-up" style={{ animationDelay: '0.2s' }}>
      <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-white/40">
        Net Liquidity Proxy
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={rows} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <defs>
            <linearGradient id="liqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#a78bfa" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="time"
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
            tickFormatter={(v) => new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'rgba(255,255,255,0.25)', fontSize: 10 }}
            tickFormatter={(v) => formatNumber(v, 0)}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<LiqTooltip />} />
          <Area
            type="monotone"
            dataKey="net_liquidity"
            stroke="#a78bfa"
            strokeWidth={2}
            fill="url(#liqGrad)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
