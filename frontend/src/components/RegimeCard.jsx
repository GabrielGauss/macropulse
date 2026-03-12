import React from 'react';
import { REGIME_CONFIG, formatDate, riskColor } from '../lib/utils';

export default function RegimeCard({ regime }) {
  if (!regime) {
    return (
      <div className="glass-card animate-pulse p-8">
        <div className="h-6 w-48 rounded bg-surface-3" />
      </div>
    );
  }

  const cfg = REGIME_CONFIG[regime.macro_regime] || REGIME_CONFIG.expansion;
  const score = regime.risk_score ?? 0;

  return (
    <div className="glass-card relative overflow-hidden p-8 animate-fade-up">
      {/* Glow accent */}
      <div
        className="pointer-events-none absolute -top-24 -right-24 h-48 w-48 rounded-full blur-[80px] opacity-30"
        style={{ background: cfg.color }}
      />

      <div className="relative z-10">
        <p className="mb-1 text-xs font-medium uppercase tracking-widest text-white/40">
          Current Macro Regime
        </p>
        <div className="mt-3 flex items-center gap-4">
          <span className="text-4xl">{cfg.icon}</span>
          <div>
            <h2 className="font-display text-3xl font-bold" style={{ color: cfg.color }}>
              {cfg.label}
            </h2>
            <p className="mt-1 text-sm text-white/50">{formatDate(regime.timestamp)}</p>
          </div>
        </div>

        {/* Risk gauge */}
        <div className="mt-6">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-white/40">Risk Score</span>
            <span className="font-mono text-2xl font-bold" style={{ color: riskColor(score) }}>
              {score > 0 ? '+' : ''}{score}
            </span>
          </div>
          <div className="mt-2 h-2 rounded-full bg-surface-3 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${Math.abs(score) / 2 + 50}%`,
                marginLeft: score < 0 ? `${50 - Math.abs(score) / 2}%` : '50%',
                background: riskColor(score),
              }}
            />
          </div>
          <div className="mt-1 flex justify-between text-[10px] text-white/25">
            <span>-100 Risk-Off</span>
            <span>0</span>
            <span>+100 Expansion</span>
          </div>
        </div>

        {/* Probability bars */}
        <div className="mt-6 grid grid-cols-2 gap-3">
          {Object.entries(regime.probabilities || {}).map(([key, val]) => {
            const rc = REGIME_CONFIG[key];
            if (!rc) return null;
            return (
              <div key={key} className="rounded-xl bg-surface-2/60 px-4 py-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-white/50">{rc.label}</span>
                  <span className="font-mono text-sm font-semibold" style={{ color: rc.color }}>
                    {(val * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 h-1.5 rounded-full bg-surface-3">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${val * 100}%`, background: rc.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {regime.volatility_state && (
          <div className="mt-4 text-xs text-white/30">
            Volatility: <span className="text-white/60">{regime.volatility_state}</span>
            {regime.model_version && (
              <span className="ml-4">Model: <span className="text-white/60">{regime.model_version}</span></span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
