import type { ReactNode } from 'react';
import { NavLink } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { PlusSquareOutlined, UnorderedListOutlined, SplitCellsOutlined, ExperimentOutlined, SafetyCertificateOutlined, ExportOutlined } from '@ant-design/icons';
import { api } from './lib/api';
import type { Capabilities } from './lib/api';

const sections = [
  { to: '/', label: '新建处理', icon: <PlusSquareOutlined /> },
  { to: '/history', label: '处理任务', icon: <UnorderedListOutlined /> },
  { to: '/compare', label: '结果对比', icon: <SplitCellsOutlined /> },
  { to: '/testing', label: '测试记录', icon: <ExperimentOutlined /> },
  { to: '/models', label: '模型验收', icon: <SafetyCertificateOutlined /> },
  { to: '/publish', label: '报告与分享', icon: <ExportOutlined /> },
];
export function Shell({ children }: { children: ReactNode }) {
  const caps = useQuery({ queryKey: ['capabilities'], queryFn: () => api<Capabilities>('/capabilities'), refetchInterval: 10000 });
  return <div className="studio"><a className="skip" href="#main">跳到工作区</a>
    <aside className="sidebar"><div className="brand"><div className="window-mark" aria-hidden="true"><i/><i/><i/><i/></div><div><strong>SeedVR2</strong><span>STUDIO / LOCAL</span></div></div>
      <div className="workspace-caption">工作区</div><nav aria-label="工作区导航">{sections.map(s => <NavLink end={s.to === '/'} key={s.to} to={s.to}>{s.icon}<span>{s.label}</span></NavLink>)}</nav>
      <div className="sidebar-foot"><span className="connection"><i className={caps.data ? 'connected' : ''}/>{caps.data ? '本地服务已连接' : caps.isPending ? '正在连接本地服务' : '本地服务未连接'}</span><div>CLI 可独立使用</div><code>{caps.data?.build ?? '版本读取中'}</code></div>
    </aside>
    <div className="content-wrap"><div className="topbar"><span>本地工作台<span className="topbar-divider">/</span><span className="muted">图片、短片与模型测试</span></span><span className="engine-state"><i/>{caps.data ? (caps.data.ncnn_linked ? 'ncnn 已接入 · 模型未验收' : 'ncnn 未接入 · 模型未验证') : '正在读取引擎状态'}</span></div>
      <main id="main" tabIndex={-1}>{children}</main><footer>文件与记录保存在本机<span>SeedVR2 → ncnn · Vulkan</span></footer>
    </div>
  </div>;
}
