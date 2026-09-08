const $ = id => document.getElementById(id);
const state = { session: null, capabilities: null, request: null, result: null, revision: 0, pending: null, imageURL: null, imageRevision: 0 };
const labels = { start: '开始处理', queue: '任务队列', compare: '结果对比', testing: '测试与报告', models: '模型与设备', publish: '发布材料' };
const sections = {
  queue: { eyebrow: 'JOBS', title: '让每个任务都有去处。', subtitle: '后续接入持久队列、取消与恢复。', current: '当前还没有运行中的推理任务。队列执行尚未实现。', items: ['任务状态由本地服务持有，刷新页面只重新读取状态。', '重试创建新的运行记录，保留之前的失败原因。', '按图片或已定义的短片段恢复，不承诺任意计算步骤续跑。'] },
  compare: { eyebrow: 'COMPARE & ANNOTATE', title: '看清变化，记录具体位置。', subtitle: '同尺度比较、逐帧查看与区域标注。', current: '当前尚未产生模型结果，对比功能等待真实媒体管线接入。', items: ['输入与输出对齐显示，视频按时间戳定位。', '记录文字变化、颜色、纹理、闪烁等问题和所在区域。', '没有参考或未计算的质量指标保持“未提供 / 未测试”。'] },
  testing: { eyebrow: 'TEST & REPRODUCE', title: '一次尝试，一份可以核对的记录。', subtitle: '测试与日常使用共用同一套应用。', current: '当前可以在开始处理页保存规划配置与真实 C++ 规划结果。模型一致性、质量与性能测试尚未接入。', items: ['规划配置可交给 CLI 重新检查，输出语义完全相同。', '后续每个测试记录输入、模型、实际设置、结果和失败原因。', '离线报告与意见关联同一次运行，未测试项目保持独立。'] },
  models: { eyebrow: 'MODELS & DEVICES', title: '先知道这台电脑能做什么。', subtitle: '每个选项都应有真实能力作为依据。', current: '以下信息来自当前本地 C++ 服务。尚未载入模型或探测显卡。', items: ['导入模型时检查清单、文件哈希和接口版本。', '设备支持、模型精度与可用内存分别检查。', '显存不足时明确说明，不自动缩小画面或更换模型。'] },
  publish: { eyebrow: 'SHARE WITH EVIDENCE', title: '让分享有依据，也方便复现。', subtitle: '在本地整理 Discussion 草稿与测试附件。', current: '当前只有框架与规划能力，尚无模型输出或性能结论。发布材料生成器等待真实运行记录接入。', items: ['从有证据的运行记录提取支持矩阵、样例和测量结果。', '保留失败、限制与未测试项，不用规划通过代替模型验收。', '先预览和导出本地材料，再由使用者决定如何公开。'] }
};

function showView(view) {
  if (!labels[view]) return;
  document.querySelectorAll('nav [data-view]').forEach(button => {
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('breadcrumb').textContent = `工作区 / ${labels[view]}`;
  $('view-start').hidden = view !== 'start';
  $('view-detail').hidden = view === 'start';
  if (view === 'start') return;
  const item = sections[view];
  for (const key of ['eyebrow', 'title', 'subtitle', 'current']) $('detail-' + key).textContent = item[key];
  $('detail-status').textContent = view === 'models' ? '服务能力已接入 · GPU 未检测' : '功能设计 · 待接入';
  $('detail-items').replaceChildren(...item.items.map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
  $('capability-details').hidden = view !== 'models';
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));

function clearResult(message = '设置已改变，请重新检查') {
  state.revision++;
  state.pending?.abort();
  state.pending = null;
  state.result = null;
  state.request = null;
  $('save-request').disabled = true;
  $('save-result').disabled = true;
  $('plan-button').disabled = !state.session || !state.capabilities?.planning;
  $('plan-button').textContent = '检查处理计划 →';
  $('plan-status').textContent = message;
  $('logical').textContent = '— × —';
  $('logical-frames').textContent = '还没有当前设置的计算结果';
  for (const id of ['working', 'windows', 'tokens']) $(id).textContent = '待计算';
  $('padding').textContent = '补边：待计算';
  $('hidden-memory').textContent = '单份中间数据：待计算';
  $('result-json').textContent = '尚未计算。';
  $('error').hidden = true;
}
function syncKind() {
  $('frames').disabled = $('kind').value === 'image';
  if ($('frames').disabled) $('frames').value = '1';
}
for (const id of ['kind', 'width', 'height', 'frames', 'scale']) $(id).addEventListener('input', () => { syncKind(); clearResult(); });
function preset(kind) {
  $('kind').value = kind;
  $('width').value = '640'; $('height').value = '360';
  $('frames').value = kind === 'image' ? '1' : '17'; $('scale').value = '2';
  state.imageRevision++;
  if (state.imageURL) URL.revokeObjectURL(state.imageURL);
  state.imageURL = null;
  $('input-preview').removeAttribute('src'); $('input-preview').hidden = true;
  $('image-file').value = '';
  $('file-hint').textContent = '当前使用声明尺寸示例，没有读取实际媒体文件。';
  syncKind(); clearResult('示例已载入，等待检查');
}
$('image-preset').addEventListener('click', () => preset('image'));
$('video-preset').addEventListener('click', () => preset('video'));
$('image-file').addEventListener('change', async () => {
  const file = $('image-file').files[0];
  if (!file) return;
  const revision = ++state.imageRevision;
  clearResult('正在读取图片尺寸…');
  if (state.imageURL) URL.revokeObjectURL(state.imageURL);
  state.imageURL = null;
  $('input-preview').hidden = true;
  if (!['image/png', 'image/jpeg'].includes(file.type) || file.size > 20 * 1024 * 1024) {
    $('file-hint').textContent = '请选择不超过 20 MiB 的 PNG 或 JPEG。';
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
    if (revision !== state.imageRevision) { URL.revokeObjectURL(url); return; }
    $('kind').value = 'image'; $('width').value = String(img.naturalWidth); $('height').value = String(img.naturalHeight);
    $('input-preview').src = url; $('input-preview').hidden = false;
    state.imageURL = url;
    $('file-hint').textContent = `${file.name} · ${img.naturalWidth} × ${img.naturalHeight} · 浏览器预览尺寸，文件未发送。`;
    syncKind(); clearResult('图片尺寸已读取，等待检查');
  } catch {
    URL.revokeObjectURL(url);
    if (revision === state.imageRevision) $('file-hint').textContent = '这张图片无法在浏览器中读取。可以手动填写尺寸。';
  }
});

let connectionRevision = 0;
async function connect() {
  const revision = ++connectionRevision;
  state.session = null;
  clearResult('等待本地服务');
  $('connection').textContent = '正在连接本地服务…';
  try {
    const options = { cache: 'no-store', signal: AbortSignal.timeout(8000) };
    const [sessionResponse, capabilityResponse] = await Promise.all([fetch('/api/v1/session', options), fetch('/api/v1/capabilities', options)]);
    if (!sessionResponse.ok || !capabilityResponse.ok) throw new Error('Connection rejected');
    const [session, capabilities] = await Promise.all([sessionResponse.json(), capabilityResponse.json()]);
    if (revision !== connectionRevision) return;
    if (!session.session_token || capabilities.schema_version !== '1.0') throw new Error('Protocol mismatch');
    state.session = session.session_token;
    state.capabilities = capabilities;
    $('capabilities-json').textContent = JSON.stringify(capabilities, null, 2);
    $('connection').textContent = '本地服务已连接';
    $('plan-button').disabled = !capabilities.planning;
    $('plan-status').textContent = '等待检查';
  } catch {
    if (revision !== connectionRevision) return;
    state.capabilities = null;
    $('capabilities-json').textContent = '服务未连接，无法读取能力。';
    $('connection').textContent = '本地服务未连接';
    $('error').textContent = '无法连接本地程序，请确认程序仍在运行，然后重新连接。';
    $('error').hidden = false;
  }
}
$('reconnect').addEventListener('click', connect);

$('plan-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!$('plan-form').reportValidity() || !state.session) return;
  clearResult('正在由 C++ 计算…');
  const revision = state.revision;
  const request = { schema_version: '1.0', model_id: 'seedvr2-3b', geometry_policy: 'preserve-content-v1', media: { kind: $('kind').value, width: Number($('width').value), height: Number($('height').value), frames: Number($('frames').value) }, output: { scale: { numerator: Number($('scale').value), denominator: 1 } } };
  const controller = new AbortController();
  state.pending = controller;
  const timeout = setTimeout(() => controller.abort(), 20000);
  $('plan-button').disabled = true;
  $('plan-button').textContent = '正在检查…';
  try {
    const response = await fetch('/api/v1/plan', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-SeedVR2-Session': state.session }, body: JSON.stringify(request), signal: controller.signal });
    const result = await response.json();
    if (revision !== state.revision) return;
    if (!response.ok) throw new Error(`${result.error?.code ?? response.status}：${result.error?.message ?? '处理计划被拒绝'}`);
    if (result.document_type !== 'geometry-plan' || result.schema_version !== '1.0') throw new Error('返回的规划格式不兼容');
    state.request = request; state.result = result;
    const output = result.logical_output, work = result.working_extent, padding = result.padding;
    $('logical').textContent = `${output.width} × ${output.height}`;
    $('logical-frames').textContent = `${output.frames} 帧 · 保持内容`;
    $('working').textContent = `${work.width} × ${work.height} · ${work.frames} 帧`;
    $('windows').textContent = `${result.windows[0].window_count} / ${result.windows[1].window_count}`;
    $('tokens').textContent = result.video_tokens.toLocaleString('zh-CN');
    $('padding').textContent = `计划补边：右 ${padding.right} / 下 ${padding.bottom} 像素 / 尾部 ${padding.tail_frames} 帧`;
    $('hidden-memory').textContent = `单份中间数据：${(result.memory.single_hidden_fp16_bytes / 1048576).toFixed(2)} MiB`;
    $('result-json').textContent = JSON.stringify(result, null, 2);
    $('plan-status').textContent = '规划已计算 · C++';
    $('save-request').disabled = false; $('save-result').disabled = false;
  } catch (error) {
    if (revision !== state.revision) return;
    $('plan-status').textContent = '检查未完成';
    $('error').textContent = controller.signal.aborted ? '本次检查超时，请重试。' : `无法完成规划。${error.message}`;
    $('error').hidden = false;
  } finally {
    clearTimeout(timeout);
    if (revision === state.revision) {
      state.pending = null;
      $('plan-button').disabled = !state.session;
      $('plan-button').textContent = '检查处理计划 →';
    }
  }
});
function download(name, data) {
  if (!data) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
$('save-request').addEventListener('click', () => download('seedvr2-planning-request.json', state.request));
$('save-result').addEventListener('click', () => download('seedvr2-geometry-plan.json', state.result));
syncKind();
connect();
