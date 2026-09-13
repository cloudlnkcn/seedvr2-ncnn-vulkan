export type Extent = { width: number; height: number; frames: number };
export type PlanningRequest = {
  schema_version: '1.0'; model_id: 'seedvr2-3b'; geometry_policy: 'preserve-content-v1';
  media: Extent & { kind: 'image' | 'video' }; output: { scale: { numerator: number; denominator: number } };
};
export type Plan = {
  logical_output: Extent; working_extent: Extent; video_tokens: number; token_grid_thw: number[];
  windows: { mode: string; window_count: number }[];
  memory: { single_hidden_fp16_bytes: number; total_peak_bytes: null };
  runnable: false; execution_blockers: string[];
};
export type Capabilities = { build: string; planning: boolean; saved_plans: boolean; evidence_audit: boolean; inference: boolean; model_certification: boolean; ncnn_linked: boolean; operator_self_test: boolean; engine: { ncnn_commit: string; awa_cpu: boolean; awa_vulkan: boolean; vae_image_diagnostics: boolean; dit_block_diagnostics: boolean; pnnx_submodel_exports: boolean; reference_profile: string } };
export type Gate = { id: string; title: string; requirement: string; status: string; missing_artifact_roles: string[] };
export type TensorCheck = { id: string; gate_id: string; shape: number[]; diagnostic_status: string; max_abs?: number | null; rmse?: number | null; nonfinite_pairs?: number; violations?: number };
export type Audit = {
  document_type: string; model_id?: string; status: string; model_verified: false; certificate: null;
  policy_id: string; policy_sha256: string; calibration_status: string; gates: Gate[]; blockers: string[];
  scope?: Record<string, string | null>; missing_scope?: string[]; manifest_sha256?: string;
  tensor_checks?: TensorCheck[]; artifact_checks?: { id: string; role: string; status: string }[];
};
export type RecordSummary = { id: string; kind: 'plan' | 'model-audit' | 'operator-test'; created_at: string; status: string };
export type Devices = { devices: { index: number; name: string }[]; default_gpu: number };
export type OperatorTest = {
  document_type: 'operator-test'; status: string; passed: boolean; backend: string;
  ncnn_commit: string; record_id?: string; model_verified: false;
  cases: { case_id: string; passed: boolean; error?: { message: string }; execution?: { device: { name: string } | null; cpu_calls: number; vulkan_calls: number; windows: number }; comparisons?: { tensor: string; max_abs: number; rmse: number; violations: number }[] }[];
};
export type SavedPlan = { document_type: 'saved-plan'; request: PlanningRequest; plan: Plan };

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  let headers: Record<string, string> = { ...options.headers as Record<string, string> };
  if (options.method && options.method !== 'GET') {
    const session = await fetch('/api/v1/session', { cache: 'no-store' });
    if (!session.ok) throw new Error('无法建立本地会话，请刷新页面。');
    const data = await session.json() as { session_token: string };
    headers = { ...headers, 'Content-Type': 'application/json', 'X-SeedVR2-Session': data.session_token };
  }
  const response = await fetch('/api/v1' + path, { ...options, headers, cache: 'no-store' });
  const data = await response.json().catch(() => ({ error: { message: `本地请求被拒绝（${response.status}）` } }));
  if (!response.ok) throw new Error(data.error?.message ?? `本地请求失败（${response.status}）`);
  return data as T;
}
export function download(name: string, content: unknown, mime = 'application/json') {
  const blob = new Blob([typeof content === 'string' ? content : JSON.stringify(content, null, 2)], { type: mime });
  const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function readableStatus(status: string) {
  return ({ MISSING_EVIDENCE: '缺少证据', EVIDENCE_PRESENT_UNREVIEWED: '待独立复核', FAILED: '发现失败', BLOCKED: '未通过验收', PLANNED: '仅已规划', PASS: '数值诊断通过', FAIL: '数值诊断失败' } as Record<string, string>)[status] ?? status;
}
export const initialRequest: PlanningRequest = {
  schema_version: '1.0', model_id: 'seedvr2-3b', geometry_policy: 'preserve-content-v1',
  media: { kind: 'video', width: 640, height: 360, frames: 17 }, output: { scale: { numerator: 2, denominator: 1 } },
};

export type Media = { id: string; name: string; width: number; height: number; bytes: number; sha256: string; url: string; kind?: 'image'|'video'; fps?: number; duration_seconds?: number; has_audio?: boolean };
export type StoragePrecision = { dit_linear_weights?: string; other_weights?: string; activation?: string; arithmetic?: string };
export type ImageModel = { storage_precision?: StoragePrecision | null; installed: boolean; worker_available: boolean; profile: string; model_verified: false; sizes: number[]; video: boolean; max_queued: number; video_model?: { profile: string; storage_precision?: StoragePrecision | null; installed: boolean; sizes: number[]; max_frames: number } };
export type ImageSettings = { schema_version: 'seedvr2-image-job-v1'|'seedvr2-video-job-v1'; media_id: string; size: number; backend: 'cpu' | 'vulkan'; gpu: number; seed: number; max_frames?: number };
export type ImageResult = { build?: string; executable_sha256?: string; ncnn_commit: string; backend: string; total_ms: number; model_manifest_sha256: string; model_verified: false; output: { path?: string; width: number; height: number; sha256: string; frames?: number; fps?: number; timestamps_90khz?: number[] }; clip?: { truncated: boolean; decoded_frames: number; padded_frames: number; latent_frames: number }; stages?: { id: string; load_ms: number; compute_ms: number; layers: number }[] };
export type Job = { id: string; status: string; sequence: number; created_at: string; updated_at: string; input: Media; request: ImageSettings; progress: { stage: string; completed: number; total: number; elapsed_ms?: number }; result: ImageResult | null; error: { message: string; code: string } | null };
export const isActive = (job?: Job) => !!job && ['QUEUED', 'RUNNING', 'CANCELLING'].includes(job.status);
export function jobStatus(status: string) {
  return ({ QUEUED: '等待处理', RUNNING: '正在处理', CANCELLING: '正在取消', SUCCEEDED: '已完成', FAILED: '处理失败', CANCELLED: '已取消', INTERRUPTED: '已中断' } as Record<string, string>)[status] ?? status;
}
export function stageName(stage?: string) {
  if (stage?.startsWith('block-')) return `修复画面细节 · ${Number(stage.slice(6)) + 1} / 32`;
  return ({ queued: '等待前面的任务完成', starting: '启动处理', validating: '核验模型文件', encoding: '分析输入画面', sampling: '准备修复', projecting: '准备画面细节', denoising: '合成修复结果', decoding: '还原图像', completed: '处理完成' } as Record<string, string>)[stage ?? ''] ?? '准备处理';
}
export const resultUrl = (id: string, name = 'output.png', save = false) => `/api/v1/jobs/${id}/files/${name}${save ? '?download=1' : ''}`;
export const isVideo = (job?: Job) => job?.request.schema_version === 'seedvr2-video-job-v1';
export const outputName = (job: Job) => isVideo(job) ? 'output.mp4' : 'output.png';
export async function uploadImage(file: File): Promise<Media> {
  const video = /\.(mp4|m4v|mov|webm|mkv)$/i.test(file.name);
  if (!video && !/\.(png|jpe?g)$/i.test(file.name) || file.size > (video ? 256 : 32) * 1024 ** 2) throw new Error('图片支持 PNG/JPEG（32 MB）；视频支持 MP4/WebM/MKV（256 MB）。');
  const session = await api<{ session_token: string }>('/session');
  const response = await fetch('/api/v1/media', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Media-Kind': video ? 'video' : 'image', 'X-File-Name': encodeURIComponent(file.name), 'X-SeedVR2-Session': session.session_token }, body: file });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message ?? '图片导入失败，请重试。');
  return result as Media;
}
