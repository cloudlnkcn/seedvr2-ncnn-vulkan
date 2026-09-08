import { useState } from 'react';
import { Alert, Button, Empty, Modal, Table } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { Page, StateTag, LoadState } from '../components/Page';
import { api, download } from '../lib/api';
import type { RecordSummary, SavedPlan } from '../lib/api';

export default function History() {
  const records = useQuery({ queryKey: ['records', 'plan'], queryFn: () => api<{ items: RecordSummary[] }>('/records?kind=plan'), refetchInterval: 5000 });
  const [detail, setDetail] = useState<SavedPlan>(), [error, setError] = useState('');
  async function view(id: string) { try { setDetail(await api<SavedPlan>('/records/' + id)); setError(''); } catch(e) { setError((e as Error).message); } }
  return <Page eyebrow="PLANS & RUNS" title="计划与任务" description="保存可重开的处理设置。当前记录为规划，尚未提交模型执行。" actions={<Link to="/planning"><Button type="primary">新建处理</Button></Link>}>
    <LoadState pending={records.isPending} error={records.error}/>{error && <Alert type="error" title={error}/>}
    <section className="panel table-panel"><Table<RecordSummary> dataSource={records.data?.items ?? []} rowKey="id" pagination={{ pageSize: 10, hideOnSinglePage: true }} scroll={{ x: 650 }} locale={{ emptyText: <Empty description="还没有保存的计划"><Link to="/planning"><Button>生成第一个规划</Button></Link></Empty> }} columns={[
      { title: '计划编号', dataIndex: 'id', render: (id: string) => <code title={id}>{id.slice(0,12)}</code> },
      { title: '保存时间', dataIndex: 'created_at', render: (v: string) => new Date(v).toLocaleString() },
      { title: '状态', dataIndex: 'status', render: (v: string) => <StateTag status={v}/> },
      { title: '操作', render: (_, r) => <div className="button-row compact"><Button size="small" onClick={() => view(r.id)}>查看</Button><Link to={'/planning?record=' + r.id}><Button size="small">重开配置</Button></Link></div> },
    ]}/></section>
    <p className="field-note">SQLite 保存最近记录列表，重启服务后保留。当前最多展示最近 100 条。执行队列、取消和断点恢复等待推理 worker 接入。</p>
    <Modal title="已保存的规划" open={!!detail} onCancel={() => setDetail(undefined)} footer={<Button onClick={() => download('saved-plan.json', detail)}>导出记录</Button>} width={720}><pre className="json-view">{JSON.stringify(detail, null, 2)}</pre></Modal>
  </Page>;
}
