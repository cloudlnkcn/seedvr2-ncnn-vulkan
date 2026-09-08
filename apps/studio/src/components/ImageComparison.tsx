import { useState } from 'react';
import { Segmented, Slider } from 'antd';

export function ImageComparison({ before, after, width, height }: { before: string; after: string; width: number; height: number }) {
  const [position, setPosition] = useState(50);
  const [mode, setMode] = useState('滑动对比');
  const [zoom, setZoom] = useState('适应窗口');
  return <div className="image-comparison">
    <div className="comparison-toolbar"><Segmented aria-label="对比方式" options={['滑动对比', '输入', '结果']} value={mode} onChange={v => setMode(String(v))}/><Segmented aria-label="显示比例" options={['适应窗口', '100%']} value={zoom} onChange={v => setZoom(String(v))}/></div>
    <div className="image-viewport"><div className={`wipe-frame ${zoom === '100%' ? 'pixel-view' : ''}`} style={{ aspectRatio: `${width}/${height}`, ...(zoom === '100%' ? { width, minWidth: width } : {}) }}>
      <img src={after} alt="SeedVR2 修复结果" draggable={false}/>
      {mode !== '结果' && <img src={before} alt="与输出对齐的输入画面" draggable={false} className="wipe-input" style={{ clipPath: `inset(0 ${mode === '输入' ? 0 : 100-position}% 0 0)` }}/>} 
      {mode === '滑动对比' && <><span className="wipe-rule" style={{ left: `${position}%` }}><i>↔</i></span><span className="image-label input-label">输入</span><span className="image-label output-label">修复结果</span></>}
    </div></div>
    {mode === '滑动对比' && <Slider ariaLabelForHandle="输入与结果分界线" value={position} onChange={setPosition} tooltip={{ formatter: null }}/>} 
    <div className="comparison-caption"><span>{width} × {height} · PNG</span><span>输入已按相同尺寸对齐，拖动分界线检查细节</span></div>
  </div>;
}
