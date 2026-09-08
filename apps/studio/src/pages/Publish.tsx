import { useState } from 'react';
import { Alert, Button, Select } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import { Page, LoadState } from '../components/Page';
import { api, download, isVideo } from '../lib/api';
import type { Audit, Capabilities, OperatorTest, RecordSummary, Job } from '../lib/api';

export default function Publish() {
  const model = useQuery({ queryKey: ['model-status'], queryFn: () => api<Audit>('/models/status') });
  const caps = useQuery({ queryKey: ['capabilities'], queryFn: () => api<Capabilities>('/capabilities') });
  const records = useQuery({ queryKey: ['records', 'operator-test'], queryFn: () => api<{ items: RecordSummary[] }>('/records?kind=operator-test') });
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => api<{ items: Job[] }>('/jobs') });
  const [jobId, setJobId] = useState<string>();
  const imageRun = useQuery({ queryKey: ['job', jobId], queryFn: () => api<Job>('/jobs/' + jobId), enabled: !!jobId });
  const [recordId, setRecordId] = useState<string>();
  const record = useQuery({ queryKey: ['operator-test', recordId], queryFn: () => api<OperatorTest>('/records/' + recordId), enabled: !!recordId });
  const ready = model.data && caps.data && (!recordId || record.data) && (!jobId || imageRun.data);
  const evidence = recordId && record.data ? `## Selected local diagnostic record

Record: ${recordId}
Backend: ${record.data.backend}
ncnn commit used by this record: ${record.data.ncnn_commit}
AWA diagnostic verdict: ${record.data.status}

${record.data.cases.map(c => `- ${c.case_id}: ${c.passed ? 'PASS' : 'FAIL'}${c.error ? ` (${c.error.message})` : ''}`).join('\n')}

This embedded self-test uses synthetic projected QKV tensors. It does not run the complete SeedVR2 model. Export the original JSON record alongside this draft to retain hashes, device details, backend calls and numerical comparisons.

` : '';
  const imageEvidence = imageRun.data ? `## Selected media task

Job: ${imageRun.data.id}
Status: ${imageRun.data.status}
Input: ${imageRun.data.input.name} (${imageRun.data.input.width} x ${imageRun.data.input.height})
Backend: ${imageRun.data.request.backend}; seed: ${imageRun.data.request.seed}
${imageRun.data.result ? `Output: ${imageRun.data.result.output.width} x ${imageRun.data.result.output.height}; total wall time: ${(imageRun.data.result.total_ms/1000).toFixed(1)} seconds, including model hashing/loading and transfers.
Output SHA-256: ${imageRun.data.result.output.sha256}
Model package SHA-256: ${imageRun.data.result.model_manifest_sha256}
Recorded ncnn commit: ${imageRun.data.result.ncnn_commit}
Recorded worker build: ${imageRun.data.result.build ?? "not recorded"}
Worker executable SHA-256: ${imageRun.data.result.executable_sha256 ?? "not recorded"}` : `Failure: ${imageRun.data.error?.message ?? 'No completed result'}`}

This is a recorded local run, not a speed benchmark or a model certificate. It uses ${isVideo(imageRun.data) ? `${imageRun.data.result?.output.frames ?? imageRun.data.request.max_frames} video frames, a whole-clip causal VAE and 3D adaptive windows` : 'one image frame'}, FP32, one Euler endpoint, CFG 1 and fixed positive conditioning. The application uses a versioned native noise generator; same-seed identity with PyTorch is not claimed.

` : '';
  const draft = ready ? `# SeedVR2 → ncnn / Vulkan: image and short-video preview with local Studio

Build: ${caps.data.build}
Pinned official ncnn commit: ${caps.data.engine.ncnn_commit}
Complete single-image restoration: implemented (preview, output long side up to 512). Short-video restoration: implemented for the first 1..17 frames, output long side up to 128, no audio or streaming cache. Model certification: not issued.

## Available implementation

- Local React / TypeScript / Ant Design workspace served by Drogon; independent C++ CLI.
- Complete native image pipeline: VAE encoder, posterior sampling/scaling, patch projections, all 32 DiT blocks, Euler endpoint and VAE decoder. Weights are loaded one block at a time.
- Isolated worker, SQLite job/event journal, bounded queue, cancellation and restart recovery.
- Image/video import, aligned image comparison, synchronized video players with frame stepping, PNG and MP4 export.
- Temporal VAE with preserved 3D kernels, framewise normalization/attention, temporal pixel shuffle, and native Vulkan dispatch.
- ncnn linked: ${caps.data.ncnn_linked}. AWA CPU: ${caps.data.engine.awa_cpu}; AWA Vulkan: ${caps.data.engine.awa_vulkan}.
- Embedded offline AWA self-test: ${caps.data.operator_self_test}; real local device detection and saved numerical reports.
- Single-frame VAE diagnostics: ${caps.data.engine.vae_image_diagnostics}; complete individual DiT block diagnostics: ${caps.data.engine.dit_block_diagnostics}.
- pnnx submodel export tools: ${caps.data.engine.pnnx_submodel_exports}. Reference profile: ${caps.data.engine.reference_profile}, adapted from pinned official code with FP32 math attention; not the official BF16/FlashAttention execution profile.
- Exported graphs and input tensors are hashed; submodel execution explicitly dispatches each layer to the requested backend.
- Model evidence auditing recomputes file SHA-256 and raw tensor differences. Diagnostic PASS never issues a model certificate.

${imageEvidence}${evidence}## Model acceptance

Policy: ${model.data.policy_id}
Policy SHA-256: ${model.data.policy_sha256}
Calibration: ${model.data.calibration_status}

${model.data.gates.map(g => `- ${g.id} ${g.title}: ${g.status}`).join('\n')}

## Retained numerical checks

On 2026-09-08, the retained 17-frame 128-pixel synthetic clip passed all 73 FP32-B checkpoints on both CPU and Vulkan after the attention and CPU encoder fixes. The original Vulkan 60/73 and CPU 70/73 failures remain archived. The official reference, shared noise and historical tolerances were not changed. See docs/VIDEO-NUMERICS.md for exact implementation identities, affected regressions and costs. These development results do not certify natural-video quality or the official BF16/FlashAttention execution path.

## Remaining work

- Streaming causal cache, video chunk boundaries, audio retention and long-video consistency.
- FP16/BF16 accuracy calibration, representative quality checks, measured memory budgets and performance profiling.
- Larger image sizes, additional image metadata handling and portable worker adapters.
- Complete evidence bundle, frozen acceptance thresholds and platform qualification.

This local draft distinguishes compiled capabilities from the selected actual test record. Complete-image diagnostics are documented in docs/image-validation.md. Historical submodel reports are in docs/awa-export-validation.json, docs/vae-image-validation.json and docs/dit-block-validation.json; review their exact scope before attaching them. Only an explicitly selected media run contributes timing. No memory benchmark or image-quality acceptance is claimed. Downloading this draft does not publish it.
` : '';
  return <Page eyebrow="REPORTS & DISCUSSION" title="把真实进展整理成可讨论的材料" description="从当前能力、实际自测记录和模型缺项生成草稿，下载后可用于 Discussion。" actions={<Button type="primary" icon={<DownloadOutlined/>} disabled={!ready} onClick={() => download('seedvr2-discussion-draft.md', draft, 'text/markdown')}>下载 Discussion 草稿</Button>}>
    <LoadState pending={model.isPending || caps.isPending || (!!recordId && record.isPending)} error={model.error || caps.error || records.error || record.error || jobs.error || imageRun.error}/>
    <Alert type="info" showIcon title="图片和短片完整链路已接入，完整模型认证尚未通过。" description="草稿保留已实现范围与尚缺环节。可附加本机实际自测结果；自测通过不代表整网模型通过验收。" className="page-notice"/>
    <section className="panel"><div className="panel-heading"><div><h2>附加本机测试证据</h2><p>可选择一份已保存的自测，失败结果也会如实写入草稿。</p></div></div><Select aria-label="Discussion 附加自测记录" style={{ width: '100%' }} allowClear value={recordId} onChange={setRecordId} placeholder="不附加自测记录" options={records.data?.items.map(r => ({ value: r.id, label: `${new Date(r.created_at).toLocaleString()} · ${r.status} · ${r.id.slice(0, 8)}` }))}/>{record.data && <Button className="diagnostic-actions" onClick={() => download(`awa-self-test-${recordId}.json`, record.data)}>下载所选原始报告</Button>}</section>
    <section className="panel image-evidence-picker"><div className="panel-heading"><h2>附加处理记录</h2></div><Select aria-label="Discussion 附加处理记录" style={{ width: '100%' }} allowClear value={jobId} onChange={setJobId} placeholder="不附加处理记录" options={jobs.data?.items.map(j => ({ value: j.id, label: `${j.input.name} · ${j.status} · ${j.id.slice(0,8)}` }))}/>{imageRun.data && <Button className="diagnostic-actions" onClick={() => download(`image-run-${jobId}.json`, imageRun.data)}>下载处理记录</Button>}</section>
    {ready && <section className="panel report-panel"><div className="panel-heading"><h2>Discussion 草稿预览</h2><span className="muted">Markdown · 本地生成</span></div><pre className="draft-view">{draft}</pre></section>}
  </Page>;
}
