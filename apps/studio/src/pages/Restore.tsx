import { useEffect, useRef, useState } from 'react';
import { Alert, Button, Collapse, Form, InputNumber, Progress, Select, Tag, Tooltip } from 'antd';
import { ArrowRightOutlined, DownloadOutlined, InboxOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { Link, useSearchParams } from 'react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Page } from '../components/Page';
import { ImageComparison } from '../components/ImageComparison';
import { VideoComparison } from '../components/VideoComparison';
import { api, download, isActive, jobStatus, resultUrl, stageName, uploadImage, outputName } from '../lib/api';
import type { Devices, ImageModel, ImageSettings, Job, Media } from '../lib/api';

export default function Restore() {
  const [search, setSearch] = useSearchParams();
  const selected = search.get('job');
  const client = useQueryClient(), input = useRef<HTMLInputElement>(null);
  const [media, setMedia] = useState<Media>();
  const [frames, setFrames] = useState(17);
  const [size, setSize] = useState(256), [seed, setSeed] = useState(666), [device, setDevice] = useState('vulkan:-1');
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [drag, setDrag] = useState(false);
  const model = useQuery({ queryKey: ['image-model'], queryFn: () => api<ImageModel>('/models/image') });
  const devices = useQuery({ queryKey: ['devices'], queryFn: () => api<Devices>('/engine/devices') });
  const job = useQuery({ queryKey: ['job', selected], queryFn: () => api<Job>('/jobs/'+selected), enabled: !!selected,
    refetchInterval: q => isActive(q.state.data) ? 1000 : false });
  const current = job.data;
  const picture = selected ? current?.input : media;
  const video = picture?.kind === 'video';
  useEffect(() => {
    if (current) { setSize(current.request.size); setFrames(current.request.max_frames ?? 17); setSeed(current.request.seed); setDevice(`${current.request.backend}:${current.request.gpu}`); }
  }, [current?.id]);
  async function importFile(file?: File) {
    if (!file || busy) return;
    setBusy(true); setError('');
    try { const added = await uploadImage(file); setMedia(added); if (added.kind === 'video') setSize(128); setSearch({}); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); if (input.current) input.current.value = ''; }
  }
  async function start() {
    if (!picture) return;
    setBusy(true); setError('');
    const [backend, gpu] = device.split(':');
    const request: ImageSettings = { schema_version: video ? 'seedvr2-video-job-v1' : 'seedvr2-image-job-v1', media_id: picture.id, size, seed, backend: backend as 'cpu' | 'vulkan', gpu: Number(gpu), ...(video ? { max_frames: frames } : {}) };
    try {
      const created = await api<Job>('/jobs', { method: 'POST', body: JSON.stringify(request) });
      client.setQueryData(['job', created.id], created); setSearch({ job: created.id });
      void client.invalidateQueries({ queryKey: ['jobs'] });
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function cancel() {
    if (!current) return;
    setBusy(true);
    try { const updated = await api<Job>(`/jobs/${current.id}/cancel`, { method: 'POST', body: '{}' }); client.setQueryData(['job', current.id], updated); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const active = isActive(current);
  const ready = (video ? model.data?.video_model?.installed : model.data?.installed) && model.data?.worker_available;
  const progress = current ? Math.min(100, Math.floor(current.progress.completed/current.progress.total*100)) : 0;
  return <Page eyebrow="IMAGE & VIDEO RESTORATION" title="让画面重新清晰" description="导入图片或短视频，选择输出尺寸，完成后直接对比与保存。关闭页面也不会中断已经开始的处理。" actions={<Link to="/history"><Button>查看全部任务</Button></Link>}>
    <input className="sr-only" type="file" accept="image/png,image/jpeg,video/mp4,video/webm,.mkv,.mov,.m4v" ref={input} onChange={e => void importFile(e.target.files?.[0])} aria-label="选择图片或视频"/>
    {(error || job.error) && <Alert className="page-notice" showIcon type="error" title={error || job.error?.message}/>}
    {!model.isPending && !ready && <Alert className="page-notice" showIcon type="warning" title="完整模型包尚未就绪" description="请在启动本地服务时指定模型包目录。已导入的图片会保留。"/>}
    <div className="restoration-layout">
      <section className="panel picture-panel">
        <div className="panel-heading"><h2>{current?.status === 'SUCCEEDED' ? '处理结果' : picture ? '输入画面' : '导入图片或视频'}</h2><Button size="small" onClick={() => input.current?.click()} disabled={busy}>{picture ? '更换文件' : '选择文件'}</Button></div>
        {current?.status === 'SUCCEEDED' && current.result ? (video ? <VideoComparison key={current.id} before={resultUrl(current.id, 'comparison-input.mp4')} after={resultUrl(current.id, 'output.mp4')} frames={current.result.output.frames} fps={current.result.output.fps} timestamps={current.result.output.timestamps_90khz}/> : <ImageComparison before={resultUrl(current.id, 'comparison-input.png')} after={resultUrl(current.id)} width={current.result.output.width} height={current.result.output.height}/>) : picture ? <div className="input-preview"><>{video ? <video src={picture.url} aria-label="待处理视频" controls muted playsInline preload="metadata"/> : <img src={picture.url} alt={picture.name}/>}</><span className="image-label input-label">输入 · {picture.width} × {picture.height}</span></div> :
          <button type="button" className={`drop-surface ${drag ? 'drag-over' : ''}`} onClick={() => input.current?.click()} disabled={busy}
            onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); void importFile(e.dataTransfer.files[0]); }}>
            <div className="drop-window"><InboxOutlined/><span className="drop-corner"/></div><strong>{busy ? '正在导入文件…' : '把图片或视频放到这里'}</strong><span>或点击选择文件</span><small>图片 32 MB / 视频 256 MB · 仅保存在本机</small>
          </button>}
        {picture && <div className="picture-meta"><strong title={picture.name}>{picture.name}</strong><span>{(picture.bytes/1024**2).toFixed(2)} MB</span></div>}
        {current?.status === 'SUCCEEDED' && <div className="button-row result-actions"><a href={resultUrl(current.id, outputName(current), true)}><Button type="primary" icon={<DownloadOutlined/>}>{video ? '保存修复视频' : '保存修复图片'}</Button></a><Link to={`/compare?job=${current.id}`}><Button>打开对比工作区</Button></Link></div>}
      </section>
      <aside className="panel restore-controls"><div className="panel-heading"><h2>处理设置</h2><Tag color={ready ? 'cyan' : 'default'}>SeedVR2 · 3B</Tag></div>
        <Form layout="vertical" onFinish={start}>
          <Form.Item label="输出长边"><Select aria-label="输出长边" value={size} disabled={active} onChange={setSize} options={(video ? [64,96,128] : [128,256,384,512]).map(v => ({ value: v, label: `${v} 像素${v === 256 ? ' · 快速测试' : v === 512 ? ' · 更多细节' : ''}` }))}/></Form.Item>
          <p className="field-note">保留画面比例，边缘会居中裁至 16 的倍数。{video ? '视频会联合处理相邻帧。' : '支持最高 512 像素长边。'}</p>
          {video && <><Form.Item label="测试片段"><Select aria-label="测试片段帧数" value={frames} disabled={active} onChange={setFrames} options={[5,9,17].map(v => ({ value: v, label: `开头 ${v} 帧${picture?.fps ? ` · 约 ${(v/picture.fps).toFixed(2)} 秒` : ''}` }))}/></Form.Item><p className="field-note">8 位 SDR 短片预览：最多 17 帧、128 像素长边，输出 MP4，不含音轨。输入不足时按实际帧数输出。</p><p className="field-note">实验性短片预览：本机保留样例已通过开发数值对照，代表性画质和长视频尚未验收。请先检查短片效果，详见<Link to="/models">模型验收</Link>。</p></>}
          <Form.Item label="处理设备"><Select aria-label="处理设备" value={device} disabled={active} onChange={setDevice} options={[{ value: 'vulkan:-1', label: '自动选择显卡' }, ...(devices.data?.devices.filter(d => !d.name.toLowerCase().includes('llvmpipe')).map(d => ({ value: `vulkan:${d.index}`, label: d.name })) ?? []), { value: 'cpu:-1', label: 'CPU · 兼容模式' }]}/></Form.Item>
          <Collapse ghost size="small" items={[{ key: 'advanced', label: '复现与高级设置', children: <><Form.Item label="随机种子"><InputNumber aria-label="随机种子" min={0} max={4294967295} precision={0} value={seed} disabled={active} onChange={v => setSeed(v ?? 666)}/></Form.Item><p className="field-note">同一版本、输入、种子与设备可复现本应用的处理。当前精度为 FP32，单步修复。</p><Button size="small" disabled={!picture} onClick={() => download('seedvr2-settings.json', { size, seed, device, model: 'seedvr2-3b' })}>导出设置</Button></> }]}/>
          <div className="start-action"><Button type="primary" htmlType="submit" size="large" block loading={busy && !active} disabled={!picture || !ready || active || job.isFetching && !current} icon={current && !active ? <ReloadOutlined/> : <ArrowRightOutlined/>}>{current && !active ? '按当前设置再处理' : '开始处理'}</Button></div>
        </Form>
        <div className="runtime-note"><i/>真实 ncnn / Vulkan 推理<span>模型文件会在每次运行前核验</span></div>
        <Link to="/models" className="quiet-link">查看模型验收状态 →</Link>
      </aside>
    </div>
    {current && <section className={`panel job-progress ${current.status === 'SUCCEEDED' ? 'finished' : ''}`} aria-live="polite">
      <div className="job-progress-title"><div><Tag color={current.status === 'SUCCEEDED' ? 'success' : current.status === 'FAILED' ? 'error' : 'processing'}>{jobStatus(current.status)}</Tag><strong>{active ? stageName(current.progress.stage) : current.error?.message ?? (current.result ? `输出 ${current.result.output.width} × ${current.result.output.height}${current.result.output.frames ? ` · ${current.result.output.frames} 帧` : ''}` : '可调整设置后重新处理')}</strong></div><div className="button-row">{active && <Button danger size="small" icon={<StopOutlined/>} loading={busy} disabled={current.status === 'CANCELLING'} onClick={cancel}>取消处理</Button>}<Tooltip title="任务记录与模型信息"><Button size="small" onClick={() => download(`seedvr2-${current.id.slice(0,8)}.json`, current)}>导出记录</Button></Tooltip></div></div>
      {active && <Progress percent={progress} showInfo={false} status="active" strokeColor="#176f70"/>}
      <div className="job-progress-foot"><span>{current.result ? `用时 ${(current.result.total_ms/1000).toFixed(1)} 秒` : current.progress.elapsed_ms ? `已处理 ${(current.progress.elapsed_ms/1000).toFixed(0)} 秒` : '任务已保存在本地'}</span><code>{current.id.slice(0,12)}</code><span>可离开此页面，在任务列表继续查看</span></div>
    </section>}
    <div className="restore-footnote"><span>图片与短片预览版。数值一致性、画质与长视频连续性分别验收；当前尚未签发完整模型认证。</span><Link to="/planning">长视频与大尺寸规划 →</Link></div>
  </Page>;
}
