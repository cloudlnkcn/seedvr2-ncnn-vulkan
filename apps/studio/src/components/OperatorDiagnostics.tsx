import { useState } from 'react';
import { Alert, Button, Select, Space, Table } from 'antd';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, download } from '../lib/api';
import type { Devices, OperatorTest, RecordSummary } from '../lib/api';
import { LoadState, StateTag } from './Page';

export function OperatorDiagnostics() {
  const queryClient = useQueryClient();
  const [gpu, setGpu] = useState(-1);
  const [recordId, setRecordId] = useState<string>();
  const devices = useQuery({ queryKey: ['engine-devices'], queryFn: () => api<Devices>('/engine/devices'), enabled: false });
  const records = useQuery({ queryKey: ['records', 'operator-test'], queryFn: () => api<{ items: RecordSummary[] }>('/records?kind=operator-test'), refetchInterval: 5000 });
  const saved = useQuery({ queryKey: ['operator-test', recordId], queryFn: () => api<OperatorTest>('/records/' + recordId), enabled: !!recordId });
  const test = useMutation({
    mutationFn: (backend: 'cpu' | 'vulkan') => api<OperatorTest>('/engine/self-test', { method: 'POST', body: JSON.stringify({ backend, gpu }) }),
    onSuccess: async data => {
      if (data.record_id) { queryClient.setQueryData(['operator-test', data.record_id], data); setRecordId(data.record_id); }
      await queryClient.invalidateQueries({ queryKey: ['records', 'operator-test'] });
    },
  });
  const result = saved.data;
  const comparisons = result?.cases.flatMap(c => (c.comparisons ?? []).map(v => ({ ...v, id: c.case_id + v.tensor, window: c.case_id.endsWith('shifted') ? '偏移窗口' : '常规窗口', device: c.execution?.device?.name ?? 'CPU' }))) ?? [];
  return <section className="panel">
    <div className="panel-heading"><div><h2>检查本机推理环境</h2><p>运行内置的 AWA 数值自测，检查 ncnn、CPU 或 Vulkan 显卡是否能正确完成计算。</p></div></div>
    <Space wrap className="diagnostic-actions">
      <Button onClick={() => devices.refetch()} loading={devices.isFetching} disabled={test.isPending}>检测显卡</Button>
      <Select aria-label="测试使用的显卡" value={gpu} onChange={setGpu} style={{ minWidth: 200, maxWidth: '100%' }} disabled={test.isPending}
        options={[{ value: -1, label: '自动选择显卡' }, ...(devices.data?.devices.map(d => ({ value: d.index, label: `${d.index} · ${d.name}` })) ?? [])]}/>
      <Button onClick={() => test.mutate('cpu')} disabled={test.isPending} loading={test.isPending && test.variables === 'cpu'}>运行 CPU 自测</Button>
      <Button type="primary" onClick={() => test.mutate('vulkan')} disabled={test.isPending} loading={test.isPending && test.variables === 'vulkan'}>运行 Vulkan 自测</Button>
    </Space>
    <p className="field-note">自测使用随程序附带的两组小型合成数据，无需下载模型。结果自动保存；通过表示这些 AWA 用例正确，不代表 SeedVR2 整体模型已通过验收。</p>
    <LoadState pending={false} error={test.error ?? devices.error ?? records.error}/>
    {records.data?.items.length ? <div className="record-picker"><label htmlFor="operator-record">查看自测记录</label><Select id="operator-record" value={recordId} onChange={setRecordId} placeholder="选择一份实际运行记录" options={records.data.items.map(r => ({ value: r.id, label: `${new Date(r.created_at).toLocaleString()} · ${r.status} · ${r.id.slice(0, 8)}` }))}/></div> : null}
    <LoadState pending={!!recordId && saved.isPending} error={saved.error}/>
    {result && <>
      <div className="panel-heading"><div><StateTag status={result.status}/> <span>{result.backend} · ncnn {result.ncnn_commit.slice(0, 12)}</span></div><Button onClick={() => download(`awa-self-test-${recordId}.json`, result)}>导出自测报告</Button></div>
      {result.cases.filter(c => c.error).map(c => <Alert key={c.case_id} type="error" showIcon title="本次自测失败" description={c.error?.message} className="page-notice"/>)}
      <Table rowKey="id" dataSource={comparisons} pagination={false} size="small" scroll={{ x: 660 }} columns={[
        { title: '窗口', dataIndex: 'window' }, { title: '输出', dataIndex: 'tensor', render: (s: string) => s === 'video' ? '视频特征' : '文本特征' },
        { title: '实际设备', dataIndex: 'device' },
        { title: '最大绝对误差', dataIndex: 'max_abs', render: (n: number) => n.toExponential(4) },
        { title: 'RMSE', dataIndex: 'rmse', render: (n: number) => n.toExponential(4) },
        { title: '超差元素', dataIndex: 'violations' },
      ]}/>
      <p className="field-note">FP32 容差：atol=1e-5、rtol=1e-4。报告包含实际后端调用次数和输入、输出哈希；自测临时张量不会留在磁盘。</p>
    </>}
  </section>;
}
