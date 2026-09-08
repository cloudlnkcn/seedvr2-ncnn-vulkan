import type { ReactNode } from 'react';
import { Alert, Spin, Tag } from 'antd';
import { readableStatus } from '../lib/api';

export function Page({ eyebrow, title, description, actions, children }: { eyebrow: string; title: string; description: string; actions?: ReactNode; children: ReactNode }) {
  return <><header className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{actions && <div className="page-actions">{actions}</div>}</header>{children}</>;
}
export function StateTag({ status }: { status: string }) {
  return <Tag color={status === 'FAILED' || status === 'FAIL' ? 'error' : status === 'PLANNED' ? 'processing' : status === 'PASS' ? 'success' : 'warning'}>{readableStatus(status)}</Tag>;
}
export function LoadState({ pending, error }: { pending?: boolean; error: Error | null }) {
  if (error) return <Alert type="error" showIcon title="无法读取本地工作区" description={`${error.message} 请确认本地服务仍在运行，然后刷新页面。`} />;
  if (pending) return <div className="loading"><Spin tip="正在读取本地数据"><div className="spinner-space" /></Spin></div>;
  return null;
}
