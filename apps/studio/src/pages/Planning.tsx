import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Form, InputNumber, Segmented, Select, Tag } from 'antd';
import { ArrowRightOutlined, DownloadOutlined, SaveOutlined } from '@ant-design/icons';
import { Link, useSearchParams } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Page } from '../components/Page';
import { api, download, initialRequest } from '../lib/api';
import type { Plan, PlanningRequest, SavedPlan } from '../lib/api';

export default function Planning() {
  const [search] = useSearchParams();
  const [request, setRequest] = useState<PlanningRequest>(initialRequest);
  const [plan, setPlan] = useState<Plan>();
  const [busy, setBusy] = useState(false), [saving, setSaving] = useState(false);
  const [error, setError] = useState(''), [saved, setSaved] = useState('');
  const revision = useRef(0); const client = useQueryClient();
  const reloadId = search.get('record');
  useEffect(() => {
    if (!reloadId) return;
    let alive = true;
    const loadingRevision = ++revision.current;
    api<SavedPlan>('/records/' + encodeURIComponent(reloadId)).then(r => {
      if (alive && loadingRevision === revision.current && r.document_type === 'saved-plan') { revision.current++; setRequest(r.request); setPlan(undefined); setSaved(''); }
    }).catch(e => { if (alive) setError(String(e.message)); });
    return () => { alive = false; };
  }, [reloadId]);
  useEffect(() => () => { revision.current++; }, []);
  function change(next: PlanningRequest) { revision.current++; setRequest(next); setPlan(undefined); setError(''); setSaved(''); }
  function media(field: 'width' | 'height' | 'frames', value: number | null) {
    change({ ...request, media: { ...request.media, [field]: value ?? 0 } });
  }
  async function evaluate() {
    const current = ++revision.current; setBusy(true); setError(''); setSaved('');
    try { const result = await api<Plan>('/plan', { method: 'POST', body: JSON.stringify(request) }); if (current === revision.current) setPlan(result); }
    catch (e) { if (current === revision.current) setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function save() {
    const current = revision.current; setSaving(true); setError('');
    try {
      const result = await api<{ id: string }>('/plans', { method: 'POST', body: JSON.stringify(request) });
      if (current === revision.current) setSaved(result.id);
      await client.invalidateQueries({ queryKey: ['records', 'plan'] });
    } catch (e) { if (current === revision.current) setError((e as Error).message); }
    finally { setSaving(false); }
  }
  return <Page eyebrow="RESTORATION WORKSPACE" title="从一次可复现的处理开始" description="先核对输出尺寸与模型条件，再开始执行。图片和视频共用同一套配置。" actions={<Link to="/models"><Button>查看模型验收</Button></Link>}>
    <Alert type="info" showIcon title="此处用于视频与大尺寸规划。图片处理请打开「新建处理」。" className="page-notice" />
    <div className="planning-layout"><section className="panel settings-panel"><div className="panel-heading"><h2>处理设置</h2><Tag>SeedVR2 · 3B</Tag></div>
      <Form layout="vertical" onFinish={evaluate}>
        <Form.Item label="输入类型"><Segmented block options={[{ label: '视频', value: 'video' }, { label: '图片', value: 'image' }]} value={request.media.kind} onChange={value => { const kind = value === "image" ? "image" : "video"; change({ ...request, media: { ...request.media, kind, frames: kind === 'image' ? 1 : 17 } }); }}/></Form.Item>
        <div className="form-pair"><Form.Item label="输入宽度"><InputNumber aria-label="输入宽度" min={1} max={65536} precision={0} value={request.media.width} onChange={v => media('width', v)}/></Form.Item>
          <Form.Item label="输入高度"><InputNumber aria-label="输入高度" min={1} max={65536} precision={0} value={request.media.height} onChange={v => media('height', v)}/></Form.Item></div>
        <Form.Item label="输入帧数"><InputNumber aria-label="输入帧数" min={1} max={4097} precision={0} disabled={request.media.kind === 'image'} value={request.media.frames} onChange={v => media('frames', v)} /></Form.Item>
        <Form.Item label="放大比例"><Select aria-label="放大比例" value={request.output.scale.numerator} options={[1, 2, 3, 4].map(v => ({ value: v, label: `${v} 倍` }))} onChange={v => change({ ...request, output: { scale: { numerator: v, denominator: 1 } } })}/></Form.Item>
        <div className="field-note">当前使用声明尺寸，尚未探测媒体文件。保存的是处理计划。</div>
        <Button type="primary" htmlType="submit" block loading={busy} icon={<ArrowRightOutlined/>}>生成规划</Button>
      </Form>
      {error && <Alert type="error" title={error} showIcon className="inline-notice"/>}
    </section>
    <section className="panel preview-panel"><div className="panel-heading"><h2>输出规划</h2><Tag color={plan ? 'processing' : 'default'}>{plan ? '几何已核对' : '等待规划'}</Tag></div>
      <div className={`geometry-stage ${plan ? 'has-plan' : ''}`}><div className="geometry-frame"><div className="frame-corner a"/><div className="frame-corner b"/><span className="frame-label">{plan ? '逻辑输出尺寸' : '声明的输入尺寸'}</span><strong>{plan?.logical_output.width ?? request.media.width}<span>×</span>{plan?.logical_output.height ?? request.media.height}</strong><span>{plan?.logical_output.frames ?? request.media.frames} 帧 · {plan ? '未执行模型推理' : '生成规划后查看输出'}</span></div></div>
      <div className="plan-details"><div><span>工作尺寸</span><strong>{plan ? `${plan.working_extent.width} × ${plan.working_extent.height}` : '—'}</strong></div><div><span>时间 / 空间网格</span><strong>{plan?.token_grid_thw.join(' × ') ?? '—'}</strong></div><div><span>普通 / 移位窗口</span><strong>{plan?.windows.map(w => w.window_count).join(' / ') ?? '—'}</strong></div><div><span>单份 FP16 hidden</span><strong>{plan ? `${(plan.memory.single_hidden_fp16_bytes / 1024 ** 2).toFixed(2)} MiB` : '—'}</strong></div></div>
      <div className="field-note">单份张量大小是理论值。模型权重、VAE、临时缓冲与 GPU 总峰值还没有实测。</div>
      <div className="button-row"><Button disabled={!plan || !!saved} loading={saving} icon={<SaveOutlined/>} onClick={save}>{saved ? '已保存计划' : '保存计划'}</Button><Button disabled={!plan} icon={<DownloadOutlined/>} onClick={() => download('geometry-plan.json', plan)}>导出规划</Button><Button onClick={() => download('planning-request.json', request)}>导出 CLI 配置</Button></div>
      {saved && <p className="saved-note">已保存到本地。<Link to="/history">打开计划与任务 →</Link></p>}
    </section></div>
    <div className="execution-strip"><div><strong>当前为视频与大尺寸规划</strong><span>视频时序处理、大尺寸显存策略与对应模型验收仍待完成。单图测试请使用新建处理。</span></div><Link to="/"><Button type="primary">单图处理</Button></Link></div>
  </Page>;
}
