import { useState } from 'react';
import { Alert, Button, Empty, Select, Table } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { Page, StateTag, LoadState } from '../components/Page';
import { api, download } from '../lib/api';
import type { Audit, RecordSummary, TensorCheck } from '../lib/api';
import { OperatorDiagnostics } from '../components/OperatorDiagnostics';

export default function Testing() {
  const records = useQuery({ queryKey: ['records', 'model-audit'], queryFn: () => api<{ items: RecordSummary[] }>('/records?kind=model-audit'), refetchInterval: 5000 });
  const [id, setId] = useState<string>();
  const audit = useQuery({ queryKey: ['audit', id], queryFn: () => api<Audit>('/records/' + id), enabled: !!id });
  return <Page eyebrow="EVIDENCE & DIAGNOSTICS" title="测试有记录，结论有范围" description="从实际文件重新计算哈希与张量误差，保留失败和缺项。验收记录不会自动升级为模型证书。" actions={<Link to="/models"><Button>模型验收要求</Button></Link>}>
    <OperatorDiagnostics/>
    <section className="panel audit-howto"><div><h2>打开一份模型证据包</h2><p>在本机运行独立 CLI。它会检查清单、逐个核对文件，并比较声明的原始 FP32 张量；保存后这里可直接查看。</p></div><pre><code>seedvr2 models audit --bundle /path/to/evidence --save</code></pre><p className="field-note">证据包目录含 manifest.json 与相对路径工件。退出码 6 表示没有获得模型认证。当前 Web 暂不上传大型模型或启动验证任务。</p></section>
    <LoadState pending={records.isPending} error={records.error}/>
    {records.data?.items.length ? <>
      <div className="record-picker"><label htmlFor="audit-record">选择验收记录</label><Select id="audit-record" placeholder="按时间选择本地记录" value={id} onChange={setId} options={records.data.items.map(r => ({ value: r.id, label: `${new Date(r.created_at).toLocaleString()} · ${r.id.slice(0, 8)} · ${r.status}` }))}/></div>
      <LoadState pending={!!id && audit.isPending} error={audit.error}/>
      {audit.data && <section className="panel"><div className="panel-heading"><div><h2>证据审计结果</h2><StateTag status={audit.data.status}/></div><Button onClick={() => download(`model-audit-${id}.json`, audit.data)}>导出完整记录</Button></div>
        <Alert type="warning" showIcon title="此记录核对本地文件完整性与数值诊断，尚未认证来源和整套模型。" description={`缺少 ${audit.data.missing_scope?.length ?? 0} 项配置身份；政策阈值未冻结。`} className="page-notice"/>
        <h3>工件完整性</h3><Table dataSource={audit.data.artifact_checks ?? []} rowKey="id" size="small" scroll={{ x: 440 }} pagination={{ pageSize: 8, hideOnSinglePage: true }} columns={[{ title: '工件', dataIndex: 'id' }, { title: '证据类型', dataIndex: 'role' }, { title: '核对结果', dataIndex: 'status' }]}/>
        <h3>原始张量比较</h3><Table<TensorCheck> dataSource={audit.data.tensor_checks ?? []} rowKey="id" size="small" scroll={{ x: 660 }} pagination={{ pageSize: 8, hideOnSinglePage: true }} columns={[
          { title: '张量', dataIndex: 'id' }, { title: '模型环节', dataIndex: 'gate_id' },
          { title: '诊断', dataIndex: 'diagnostic_status', render: (s: string) => <StateTag status={s}/> },
          { title: '最大绝对误差', dataIndex: 'max_abs', render: (n?: number | null) => n == null ? '—' : n.toExponential(4) },
          { title: 'RMSE', dataIndex: 'rmse', render: (n?: number | null) => n == null ? '—' : n.toExponential(4) },
          { title: '非有限值', dataIndex: 'nonfinite_pairs' },
        ]}/><p className="field-note">FP32 单算子诊断使用 atol=1e-5、rtol=1e-4。此容差不用于 BF16 / FP16 整网验收。原始报告保留每项缺失证据、哈希与配置身份。</p>
      </section>}
    </> : records.data && <section className="panel empty-panel"><Empty description="还没有保存的模型验收记录"/><p>准备官方参考和候选输出后，执行上面的命令。合成数据用于测试验收工具，不能当作模型证据。</p></section>}
  </Page>;
}
