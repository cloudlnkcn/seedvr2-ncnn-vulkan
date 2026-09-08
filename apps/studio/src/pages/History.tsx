import { useState } from 'react';
import { Alert, Button, Empty, Table, Tag } from 'antd';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router';
import { Page, LoadState } from '../components/Page';
import { api, isActive, jobStatus, resultUrl, stageName, isVideo, outputName } from '../lib/api';
import type { Job } from '../lib/api';

export default function History() {
  const client = useQueryClient(), [error, setError] = useState('');
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => api<{ items: Job[] }>('/jobs'), refetchInterval: 2000 });
  async function cancel(id: string) {
    try { await api(`/jobs/${id}/cancel`, { method: 'POST', body: '{}' }); await client.invalidateQueries({ queryKey: ['jobs'] }); }
    catch(e) { setError((e as Error).message); }
  }
  return <Page eyebrow="PROCESSING HISTORY" title="处理任务" description="任务与结果保存在本机。可以依次提交图片或短片，工作台会按顺序处理，页面关闭后继续运行。" actions={<Link to="/"><Button type="primary">新建处理</Button></Link>}>
    <LoadState pending={jobs.isPending} error={jobs.error}/>{error && <Alert type="error" title={error}/>}
    <section className="panel table-panel"><Table<Job> dataSource={jobs.data?.items ?? []} rowKey="id" pagination={{ pageSize: 10, hideOnSinglePage: true }} scroll={{ x: 850 }} locale={{ emptyText: <Empty description="还没有处理任务"><Link to="/"><Button>导入图片或视频</Button></Link></Empty> }} columns={[
      { title: '文件', key: 'input', render: (_, r) => <div className="history-picture">{isVideo(r) && r.status !== 'SUCCEEDED' ? <span className="history-video-label">视频</span> : <img src={r.status === 'SUCCEEDED' ? resultUrl(r.id) : r.input.url} alt="" loading="lazy"/>}<div><strong title={r.input.name}>{r.input.name}</strong><span>{r.input.width} × {r.input.height} → 长边 {r.request.size}</span></div></div> },
      { title: '状态', dataIndex: 'status', width: 180, render: (s: string, r) => <><Tag color={s === 'SUCCEEDED' ? 'success' : s === 'FAILED' ? 'error' : isActive(r) ? 'processing' : 'default'}>{jobStatus(s)}</Tag>{isActive(r) && <div className="table-subtext">{stageName(r.progress.stage)}</div>}</> },
      { title: '创建时间', dataIndex: 'created_at', render: (v: string) => <span className="table-subtext">{new Date(v).toLocaleString()}</span> },
      { title: '操作', width: 195, render: (_, r) => <div className="button-row compact"><Link to={`/?job=${r.id}`}><Button size="small">{isActive(r) ? '查看进度' : r.status === 'SUCCEEDED' ? '查看结果' : '重开设置'}</Button></Link>{r.status === 'SUCCEEDED' ? <a href={resultUrl(r.id, outputName(r), true)}><Button size="small">保存</Button></a> : isActive(r) ? <Button size="small" danger disabled={r.status === 'CANCELLING'} onClick={() => cancel(r.id)}>取消</Button> : null}</div> },
    ]}/></section>
    <p className="field-note">最多同时保留 8 个待处理任务。服务关闭或意外退出后，未完成的任务会标记为中断，可重开设置再处理。<Link to="/plans">查看先前保存的几何计划 →</Link></p>
  </Page>;
}
