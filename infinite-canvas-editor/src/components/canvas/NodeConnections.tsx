import type { CanvasNode, Connection } from '../../types/canvas'

function portRight(n: CanvasNode) {
  return { x: n.x + n.width, y: n.y + n.height / 2 }
}

function portLeft(n: CanvasNode) {
  return { x: n.x, y: n.y + n.height / 2 }
}

type Props = {
  nodes: CanvasNode[]
  connections: Connection[]
  zoom: number
}

export function NodeConnections({ nodes, connections, zoom }: Props) {
  const map = new Map(nodes.map((n) => [n.id, n]))

  return (
    <svg
      className="pointer-events-none absolute inset-0 overflow-visible"
      style={{ width: '100%', height: '100%' }}
    >
      {connections.map((c) => {
        const a = map.get(c.from)
        const b = map.get(c.to)
        if (!a || !b) return null
        const p0 = portRight(a)
        const p3 = portLeft(b)
        const dist = Math.max(40, Math.abs(p3.x - p0.x) * 0.45)
        const p1 = { x: p0.x + dist, y: p0.y }
        const p2 = { x: p3.x - dist, y: p3.y }
        const d = `M ${p0.x} ${p0.y} C ${p1.x} ${p1.y} ${p2.x} ${p2.y} ${p3.x} ${p3.y}`
        return (
          <path
            key={c.id}
            d={d}
            fill="none"
            stroke="rgba(59,130,246,0.55)"
            strokeWidth={2 / zoom}
          />
        )
      })}
    </svg>
  )
}
