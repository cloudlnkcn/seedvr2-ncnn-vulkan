import { Button, Empty, Select } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router';
import { Page, LoadState } from '../components/Page';
import { ImageComparison } from '../components/ImageComparison';
import { VideoComparison } from '../components/VideoComparison';
import { api, resultUrl, isVideo, outputName } from '../lib/api';
import type { Job } from '../lib/api';

export default function Compare() {
  const [search, setSearch] = useSearchParams();
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: () => api<{ items: Job[] }>('/jobs') });
  const completed = jobs.data?.items.filter(j => j.status === 'SUCCEEDED') ?? [];
  const selected = completed.find(j => j.id === search.get('job')) ?? completed[0];
  return <Page eyebrow="OUTPUT REVIEW" title="看清每一处变化" description="图片可拖动分界线对比，视频可同步播放与逐帧查看。两侧输入与结果已对齐。" actions={selected && <a href={resultUrl(selected.id, outputName(selected), true)}><Button type="primary">{isVideo(selected) ? '保存修复视频' : '保存修复图片'}</Button></a>}>
    <LoadState pending={jobs.isPending} error={jobs.error}/>
    {selected && selected.result ? <><div className="record-picker"><span>选择处理结果</span><Select aria-label="选择处理结果" value={selected.id} onChange={v => setSearch({ job: v })} options={completed.map(j => ({ value: j.id, label: `${j.input.name} · ${j.result?.output.width} × ${j.result?.output.height} · ${new Date(j.created_at).toLocaleTimeString()}` }))}/><Link to={`/?job=${selected.id}`}><Button>查看处理设置</Button></Link></div><section className="panel">{isVideo(selected) ? <VideoComparison key={selected.id} before={resultUrl(selected.id, 'comparison-input.mp4')} after={resultUrl(selected.id, 'output.mp4')} frames={selected.result.output.frames} fps={selected.result.output.fps} timestamps={selected.result.output.timestamps_90khz}/> : <ImageComparison key={selected.id} before={resultUrl(selected.id, 'comparison-input.png')} after={resultUrl(selected.id)} width={selected.result.output.width} height={selected.result.output.height}/>}</section><p className="field-note">运行用时 {(selected.result.total_ms/1000).toFixed(1)} 秒 · {selected.result.backend} · 画面对比不等同于数值一致性验收。<a href={resultUrl(selected.id, 'run.json', true)}>下载完整运行记录</a></p></> : !jobs.isPending && <section className="panel comparison-empty"><Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="完成一次图片或视频处理，即可在这里对比结果"/><Link to="/"><Button type="primary">去处理画面</Button></Link></section>}
  </Page>;
}
