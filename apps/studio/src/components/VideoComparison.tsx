import { useRef, useState } from 'react';
import { Alert, Button } from 'antd';
import { CaretRightOutlined, PauseOutlined, StepBackwardOutlined, StepForwardOutlined } from '@ant-design/icons';

export function VideoComparison({ before, after, frames = 1, fps = 8, timestamps }: { before: string; after: string; frames?: number; fps?: number; timestamps?: number[] }) {
  const left = useRef<HTMLVideoElement>(null), right = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false), [frame, setFrame] = useState(0), [error, setError] = useState('');
  const [loaded, setLoaded] = useState(0);
  const times = timestamps?.map(t => t/90000) ?? Array.from({ length: frames }, (_, i) => i/fps);
  const frameAt = (time: number) => {
    let index = 0; while (index+1 < times.length && times[index+1] <= time+.002) index++;
    return index;
  };
  const stop = () => { left.current?.pause(); right.current?.pause(); setPlaying(false); };
  const seek = (index: number) => {
    stop(); const next = Math.min(frames-1, Math.max(0, index)); setFrame(next);
    for (const video of [left.current, right.current]) if (video) video.currentTime = times[next];
  };
  async function play() {
    if (playing) { seek(frameAt(right.current?.currentTime ?? 0)); return; }
    if (frame >= frames-1) seek(0);
    try { await Promise.all([left.current?.play(), right.current?.play()]); setPlaying(true); }
    catch { stop(); setError('浏览器暂时无法播放，可以下载 MP4 在本机查看。'); }
  }
  return <div className="video-comparison">
    <div className="video-pair">
      <div><span className="video-label">输入片段</span><video ref={left} src={before} muted playsInline preload="auto" aria-label="输入视频" onLoadedData={() => setLoaded(v => v|1)} onError={() => setError('输入片段加载失败，请刷新后重试。')}/></div>
      <div><span className="video-label result">修复结果</span><video ref={right} src={after} muted playsInline preload="auto" aria-label="修复视频" onLoadedData={() => setLoaded(v => v|2)} onEnded={() => seek(frames-1)} onError={() => setError('修复视频加载失败，请刷新或下载后查看。')} onTimeUpdate={() => {
        const time = right.current?.currentTime ?? 0;
        setFrame(frameAt(time));
        if (left.current && Math.abs(left.current.currentTime-time) > Math.min(.08,.5/fps)) left.current.currentTime = time;
      }}/></div>
    </div>
    <div className="video-transport">
      <Button type="primary" onClick={play} disabled={loaded !== 3} icon={playing ? <PauseOutlined/> : <CaretRightOutlined/>}>{playing ? '暂停对比' : '同步播放'}</Button>
      <Button aria-label="上一帧" icon={<StepBackwardOutlined/>} onClick={() => seek(frame-1)} disabled={loaded !== 3 || frame===0}/>
      <input aria-label="视频帧时间线" type="range" min={0} max={frames-1} step={1} value={frame} onChange={e => seek(Number(e.target.value))} disabled={loaded !== 3}/>
      <Button aria-label="下一帧" icon={<StepForwardOutlined/>} onClick={() => seek(frame+1)} disabled={loaded !== 3 || frame===frames-1}/>
      <span className="video-frame">{frame+1} / {frames} 帧</span>
    </div>
    {error && <Alert type="error" showIcon title={error}/>}
    <p className="field-note">两侧时间同步。暂停后可逐帧检查细节与闪烁；当前短片输出不含音轨。</p>
  </div>;
}
