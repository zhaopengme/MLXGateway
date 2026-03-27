import { useRef } from 'react'
import { useCanvasStore } from '../../stores/canvasStore'
import { useCanvasInteraction } from '../../hooks/useCanvasInteraction'
import { CanvasNode } from './CanvasNode'
import { NodeConnections } from './NodeConnections'
import { CanvasToolbar } from './CanvasToolbar'

type Props = { isDark: boolean }

export function InfiniteCanvas({ isDark }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const nodes = useCanvasStore((s) => s.nodes)
  const connections = useCanvasStore((s) => s.connections)
  const view = useCanvasStore((s) => s.view)
  const tool = useCanvasStore((s) => s.tool)

  const { onCanvasPointerDown, onCanvasPointerMove, onCanvasPointerUp } =
    useCanvasInteraction(canvasRef)

  const gridColor = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)'

  return (
    <div
      id="infinite-canvas-view"
      ref={canvasRef}
      className={`relative flex-1 min-h-0 min-w-0 overflow-hidden select-none-canvas ${
        isDark ? 'bg-zinc-950' : 'bg-zinc-100'
      } ${tool === 'pan' ? 'cursor-grab active:cursor-grabbing' : ''}`}
      style={{
        backgroundImage: `linear-gradient(${gridColor} 1px, transparent 1px), linear-gradient(90deg, ${gridColor} 1px, transparent 1px)`,
        backgroundSize: `${24 * view.zoom}px ${24 * view.zoom}px`,
        backgroundPosition: `${view.x}px ${view.y}px`,
      }}
      onPointerDown={onCanvasPointerDown}
      onPointerMove={onCanvasPointerMove}
      onPointerUp={onCanvasPointerUp}
    >
      <CanvasToolbar isDark={isDark} />
      <div
        className="absolute top-0 left-0 w-full h-full origin-top-left will-change-transform"
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
        }}
      >
        <NodeConnections nodes={nodes} connections={connections} zoom={view.zoom} />
        {nodes.map((node) => (
          <CanvasNode key={node.id} node={node} canvasRef={canvasRef} isDark={isDark} />
        ))}
      </div>
    </div>
  )
}
