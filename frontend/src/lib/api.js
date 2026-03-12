const BASE = '';

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  getCurrentRegime: () => apiFetch('/v1/regime/current'),
  getRegimeHistory: (limit = 90) => apiFetch(`/v1/regime/history?limit=${limit}`),
  getLiquidity: (limit = 60) => apiFetch(`/v1/liquidity?limit=${limit}`),
  getFactors: (limit = 60) => apiFetch(`/v1/factors?limit=${limit}`),
  getDrift: (limit = 30) => apiFetch(`/v1/drift?limit=${limit}`),
  runBacktest: (params) =>
    apiFetch('/v1/backtest', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  getHealth: () => apiFetch('/health'),
};
