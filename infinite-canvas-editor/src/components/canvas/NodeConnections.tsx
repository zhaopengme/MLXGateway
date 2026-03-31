import { useState } from 'react'
import { useCanvasStore } from '../../stores/canvasStore'
import type { CanvasNode, Connection } from '../../types/canvas'

function portRight(n: CanvasNode) {
  return { x: n.x + n.width, y: n.y + n.height / 2 }
}

function portLeft(n: CanvasNode, c?: Connection) {
  if (c?.toPort === 'ref-0') return { x: n.x, y: n.y + n.height * 0.3 }
  if (c?.toPort === 'ref-1') return { x: n.x, y: n.y + n.height * 0.6 }
  return { x: n.x, y: n.y + n.height / 2 }
}

function bezierPath(a: CanvasNode, b: CanvasNode, c?: Connection) {
  const p0 = portRight(a)
  const p3 = portLeft(b, c)
  const dist = Math.max(40, Math.abs(p3.x - p0.x) * 0.45)
  const p1 = { x: p0.x + dist, y: p0.y }
  const p2 = { x: p3.x - dist, y: p3.y }
  return `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`
}

type Props = {
  nodes: CanvasNode[]
  connections: Connection[]
  zoom: number
}

export function NodeConnections({ nodes, connections, zoom }: Props) {
  const map = new Map(nodes.map((n) => [n.id, n]))
  const removeConnection = useCanvasStore((s) => s.removeConnection)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  return (
    <svg
      className="absolute inset-0 overflow-visible"
      style={{ width: '100%', height: '100%' }}
    >
      {connections.map((c) => {
        const a = map.get(c.from)
        const b = map.get(c.to)
        if (!a || !b) return null
        const d = bezierPath(a, b, c)
        const hovered = hoveredId === c.id
        return (
          <g key={c.id}>
            <path
              d={d}
              fill="none"
              stroke="transparent"
              strokeWidth={12 / zoom}
              style={{ cursor: 'pointer' }}
              onPointerEnter={() => setHoveredId(c.id)}
              onPointerLeave={() => setHoveredId(null)}
              onPointerDown={(e) => {
                e.stopPropagation()
                removeConnection(c.id)
              }}
            />
            <path
              d={d}
              fill="none"
              stroke={hovered ? 'rgba(239,68,68,0.8)' : 'rgba(59,130,246,0.55)'}
              strokeWidth={(hovered ? 3 : 2) / zoom}
              style={{ pointerEvents: 'none', transition: 'stroke 0.15s, stroke-width 0.15s' }}
            />
          </g>
        )
      })}
    </svg>
  )
}
