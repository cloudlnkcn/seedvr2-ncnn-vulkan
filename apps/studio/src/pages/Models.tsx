import { Alert, Button, Tag } from 'antd';
import { DownloadOutlined, ExperimentOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router';
import { LoadState, Page, StateTag } from '../components/Page';
import { api, download } from '../lib/api';
import type { Audit, Capabilities } from '../lib/api';

export default function Models() {
  const state = useQuery({ queryKey: ['model-status'], queryFn: () => api<Audit>('/models/status') });
  const caps = useQuery({ queryKey: ['capabilities'], queryFn: () => api<Capabilities>('/capabilities') });
  const model = state.data;
  return <Page eyebrow="MODEL VALIDATION" title="模型，逐项验收" description="验收覆盖模型本身的计算与输出。每份结论绑定权重、导出、设备、精度及测试范围。" actions={<Button icon={<DownloadOutlined/>} disabled={!model} onClick={() => download('model-validation-status.json', model)}>导出当前缺项</Button>}>
    <LoadState pending={state.isPending} error={state.error}/>
    {model && <><section className="model-overview"><div className="model-identity"><span className="model-wordmark">SeedVR2<span>3B</span></span><div><h2>官方基线 → ncnn / Vulkan</h2><p>单步修复 · 图像与视频 · 首个验收目标</p><Tag color="warning">{model.calibration_status === 'NOT_FROZEN' ? '验收阈值尚未冻结' : model.calibration_status}</Tag></div></div><div className="verdict"><StateTag status={model.status}/><span>模型证书未签发</span></div></section>
      <Alert showIcon type="warning" title="图片和短片可以实际处理，完整模型认证仍未通过。" description="2026-09-08 本机保留的 17 帧、128 像素合成短片，在 CPU 和 Vulkan 上各通过 73/73 项官方 FP32-B 数值对照；此前失败记录继续保留。开发对照不等于画质认证：官方 BF16 基线、代表性自然视频、长视频和更多设备仍需验收。" className="page-notice"/>
      {caps.data && <section className="panel"><div className="panel-heading"><div><h2>当前可以实际测试什么</h2><p>ncnn {caps.data.engine.ncnn_commit.slice(0, 12)} · FP32 · CPU / Vulkan</p></div><Link to="/"><Button type="primary">导入图片或短片</Button></Link></div><p>单图支持最高 512 像素长边；短片支持开头最多 17 帧、128 像素长边。视频经过时序 VAE 和全部 32 层时空注意力，按整段联合处理。可同步播放、逐帧比较并下载 MP4，当前输出不含音轨。</p><p className="field-note">开发数值对比与日常处理记录分别保留。日常处理成功只证明本次执行完成，不会自动提升下方认证状态。无需权重的 AWA 算子自测可在<Link to="/testing">测试记录</Link>中运行。</p></section>}
      <div className="section-intro"><h2>12 项模型验收</h2><span>所有必需项都有完整证据，才能进入签发流程</span></div>
      <div className="gate-grid">{model.gates.map(gate => <article className="gate" key={gate.id}><div className="gate-top"><span className="gate-id">{gate.id}</span><StateTag status={gate.status}/></div><h3>{gate.title}</h3><p>{gate.requirement}</p><div className="gate-missing">缺少 {gate.missing_artifact_roles.length} 类证据</div></article>)}</div>
      <section className="panel model-policy"><div><h2>验收策略有固定身份</h2><p>阈值校准与冻结需要先于候选模型测试。任何配置变化，都需要重新核对验收范围。</p><code>{model.policy_id}</code><code className="hash-value">SHA256 {model.policy_sha256}</code></div><Link to="/testing"><Button icon={<ExperimentOutlined/>}>查看验收记录</Button></Link></section>
    </>}
  </Page>;
}
