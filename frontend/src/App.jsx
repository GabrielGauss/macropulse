import React, { useEffect, useCallback } from 'react';
import Header from './components/Header';
import RegimeCard from './components/RegimeCard';
import RegimeTimeline from './components/RegimeTimeline';
import LiquidityChart from './components/LiquidityChart';
import FactorsChart from './components/FactorsChart';
import DriftPanel from './components/DriftPanel';
import { useFetch } from './hooks/useFetch';
import { useRegimeSocket } from './hooks/useRegimeSocket';
import { api } from './lib/api';

export default function App() {
  const { connected, lastMessage } = useRegimeSocket();

  const fetchRegime = useCallback(() => api.getCurrentRegime(), []);
  const fetchHistory = useCallback(() => api.getRegimeHistory(90), []);
  const fetchLiquidity = useCallback(() => api.getLiquidity(60), []);
  const fetchFactors = useCallback(() => api.getFactors(60), []);
  const fetchDrift = useCallback(() => api.getDrift(30), []);

  const regime = useFetch(fetchRegime);
  const history = useFetch(fetchHistory);
  const liquidity = useFetch(fetchLiquidity);
  const factors = useFetch(fetchFactors);
  const drift = useFetch(fetchDrift);

  // Refresh when a WebSocket update arrives
  useEffect(() => {
    if (lastMessage) {
      regime.refetch();
      history.refetch();
      liquidity.refetch();
      factors.refetch();
      drift.refetch();
    }
  }, [lastMessage]);

  return (
    <div className="flex min-h-screen flex-col">
      <Header connected={connected} />

      <main className="flex-1 p-6">
        {regime.error && (
          <div className="mb-4 rounded-xl border border-accent-amber/30 bg-accent-amber/10 px-4 py-3 text-sm text-accent-amber">
            Unable to connect to API. Make sure the backend is running.
          </div>
        )}

        <div className="mx-auto max-w-7xl space-y-6">
          {/* Top row */}
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <RegimeCard regime={regime.data} />
            </div>
            <DriftPanel data={drift.data} />
          </div>

          {/* Timeline */}
          <RegimeTimeline history={history.data} />

          {/* Bottom charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <LiquidityChart data={liquidity.data} />
            <FactorsChart data={factors.data} />
          </div>

          {/* Footer */}
          <footer className="border-t border-white/[0.04] pt-4 text-center text-[10px] text-white/20">
            MacroPulse v0.1.0 — Probabilistic macro regime intelligence
          </footer>
        </div>
      </main>
    </div>
  );
}
