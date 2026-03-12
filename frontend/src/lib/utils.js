export const REGIME_CONFIG = {
  expansion: { color: '#00e5a0', bg: 'rgba(0,229,160,0.12)', label: 'Expansion', icon: '↗' },
  tightening: { color: '#ff4d6a', bg: 'rgba(255,77,106,0.12)', label: 'Tightening', icon: '↘' },
  risk_off: { color: '#ffb800', bg: 'rgba(255,184,0,0.12)', label: 'Risk-Off', icon: '⚠' },
  recovery: { color: '#3b82f6', bg: 'rgba(59,130,246,0.12)', label: 'Recovery', icon: '↺' },
};

export function formatDate(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function formatNumber(n, decimals = 1) {
  if (n == null) return '—';
  if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(decimals) + 'T';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(decimals) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(decimals) + 'M';
  return n.toFixed(decimals);
}

export function riskColor(score) {
  if (score >= 30) return '#00e5a0';
  if (score >= 0) return '#3b82f6';
  if (score >= -30) return '#ffb800';
  return '#ff4d6a';
}
