import { Component, lazy, Suspense } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider, Spin } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HashRouter, Route, Routes } from 'react-router';
import { Shell } from './shell';
import './style.css';

const Planning = lazy(() => import('./pages/Planning'));
const Restore = lazy(() => import('./pages/Restore'));
const Plans = lazy(() => import('./pages/Plans'));
const History = lazy(() => import('./pages/History'));
const Models = lazy(() => import('./pages/Models'));
const Testing = lazy(() => import('./pages/Testing'));
const Compare = lazy(() => import('./pages/Compare'));
const Publish = lazy(() => import('./pages/Publish'));
const client = new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 5000, refetchOnWindowFocus: true } } });

class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('Studio view failed', error, info.componentStack); }
  render() { return this.state.failed ? <main className="fatal"><h1>工作区页面未能打开</h1><p>已保存的记录仍在本地。重新加载页面后再试。</p><button onClick={() => location.reload()}>重新加载</button></main> : this.props.children; }
}
createRoot(document.getElementById('root')!).render(
  <ErrorBoundary><ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: '#176f70', colorInfo: '#2b61b6', colorText: '#203747', borderRadius: 8, fontFamily: '"Noto Sans CJK SC", "Microsoft YaHei", system-ui, sans-serif', controlHeight: 38 } }}>
    <QueryClientProvider client={client}><HashRouter><Shell><Suspense fallback={<div className="loading"><Spin /></div>}><Routes>
      <Route path="/" element={<Restore />} /><Route path="/history" element={<History />} />
      <Route path="/planning" element={<Planning />} /><Route path="/plans" element={<Plans />} />
      <Route path="/models" element={<Models />} /><Route path="/testing" element={<Testing />} />
      <Route path="/compare" element={<Compare />} /><Route path="/publish" element={<Publish />} />
      <Route path="*" element={<Restore />} />
    </Routes></Suspense></Shell></HashRouter></QueryClientProvider>
  </ConfigProvider></ErrorBoundary>,
);
