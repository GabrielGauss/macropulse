import React from 'react';
import { AlertTriangle, CheckCircle } from 'lucide-react';

const THRESHOLDS = {
  pca_explained_variance: 0.10,
  regime_persistence: 0.97,
  feature_mean_shift: 1.5,
  feature_std_shift: 1.5,
};

const METRIC_LABELS = {
  pca_explained_variance: 'PCA Variance Drift',
  regime_persistence: 'Regime Persistence',
  feature_mean_shift: 'Feature Mean Shift',
  feature_std_shift: 'Feature Std Shift',
};

function StatusIcon({ value, threshold }) {
  const warn = value > threshold;
  return warn
    ? <AlertTriangle className="h-4 w-4 text-accent-amber" />
    : <CheckCircle className="h-4 w-4 text-accent-green" />;
}

export default function DriftPanel({ data }) {
  if (!data?.data?.length) {
    return (
      <div className="glass-card flex h-48 items-center justify-center">
        <p className="text-sm text-white/30">No drift data</p>
      </div>
    );
  }

  const latest = data.data[0];

  return (
    <div className="glass-card p-6 animate-fade-up" style={{ animationDelay: '0.4s' }}>
      <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-white/40">
        Model Drift Monitor
      </h3>
      <div className="space-y-3">
        {Object.entries(METRIC_LABELS).map(([key, label]) => {
          const val = latest[key];
          const threshold = THRESHOLDS[key];
          if (val == null) return null;
          const pct = Math.min((val / (threshold * 2)) * 100, 100);
          const warn = val > threshold;
          return (
            <div key={key}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <StatusIcon value={val} threshold={threshold} />
                  <span className="text-xs text-white/60">{label}</span>
                </div>
                <span className={`font-mono text-xs font-semibold ${warn ? 'text-accent-amber' : 'text-accent-green'}`}>
                  {val.toFixed(4)}
                </span>
              </div>
              <div className="mt-1 h-1 rounded-full bg-surface-3">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${pct}%`,
                    background: warn ? '#ffb800' : '#00e5a0',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {latest.model_version && (
        <p className="mt-4 text-[10px] text-white/25">Model: {latest.model_version}</p>
      )}
    </div>
  );
}
