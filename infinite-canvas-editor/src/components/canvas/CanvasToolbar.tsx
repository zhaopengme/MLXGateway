import { Hand, Minus, Plus, Scan } from 'lucide-react'
import { useCanvasStore } from '../../stores/canvasStore'

type Props = { isDark: boolean }

export function CanvasToolbar({ isDark }: Props) {
  const view = useCanvasStore((s) => s.view)
  const setView = useCanvasStore((s) => s.setView)
  const tool = useCanvasStore((s) => s.tool)
  const setTool = useCanvasStore((s) => s.setTool)
  const nodes = useCanvasStore((s) => s.nodes)

  const zoomFit = () => {
    if (!nodes.length) return
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity
    for (const n of nodes) {
      minX = Math.min(minX, n.x)
      minY = Math.min(minY, n.y)
      maxX = Math.max(maxX, n.x + n.width)
      maxY = Math.max(maxY, n.y + n.height)
    }
    const pad = 80
    const w = maxX - minX + pad * 2
    const h = maxY - minY + pad * 2
    const el = document.getElementById('infinite-canvas-view')
    const rw = el?.clientWidth ?? 800
    const rh = el?.clientHeight ?? 600
    const z = Math.min(rw / w, rh / h, 1.2)
    setView({
      zoom: Math.max(0.2, Math.round(z * 1000) / 1000),
      x: (rw - w * z) / 2 - minX * z + pad * z,
      y: (rh - h * z) / 2 - minY * z + pad * z,
    })
  }

  const btn = `px-2 py-1 rounded text-xs font-medium flex items-center gap-1 ${
    isDark
      ? 'bg-zinc-800 text-zinc-200 hover:bg-zinc-700 border border-zinc-700'
      : 'bg-white text-zinc-800 hover:bg-zinc-50 border border-zinc-200'
  }`

  return (
    <div className="absolute top-3 right-3 z-30 flex items-center gap-2">
      <button
        type="button"
        className={`${btn} ${tool === 'pan' ? 'ring-2 ring-blue-500' : ''}`}
        onClick={() => setTool(tool === 'pan' ? 'select' : 'pan')}
        title="Pan (or middle mouse)"
      >
        <Hand className="size-3.5" />
        Pan
      </button>
      <button type="button" className={btn} onClick={() => setView({ ...view, zoom: Math.max(0.15, view.zoom / 1.15) })}>
        <Minus className="size-3.5" />
      </button>
      <span className={`text-[11px] tabular-nums min-w-[3rem] text-center ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
        {Math.round(view.zoom * 100)}%
      </span>
      <button type="button" className={btn} onClick={() => setView({ ...view, zoom: Math.min(4, view.zoom * 1.15) })}>
        <Plus className="size-3.5" />
      </button>
      <button type="button" className={btn} onClick={zoomFit} title="Fit all nodes">
        <Scan className="size-3.5" />
        Fit
      </button>
    </div>
  )
}
