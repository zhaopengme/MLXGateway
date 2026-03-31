import { Trash2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useCanvasStore } from '../../stores/canvasStore'
import type { CanvasNode as NodeType } from '../../types/canvas'
import { useNodeDrag } from '../../hooks/useNodeDrag'
import { CanvasNodeBody } from './CanvasNodeBody'

type Props = {
  node: NodeType
  canvasRef: React.RefObject<HTMLElement | null>
  isDark: boolean
}

export function CanvasNode({ node, canvasRef, isDark }: Props) {
  const selectedNodeIds = useCanvasStore((s) => s.selectedNodeIds)
  const selectOnly = useCanvasStore((s) => s.selectOnly)
  const toggleSelect = useCanvasStore((s) => s.toggleSelect)
  const removeNode = useCanvasStore((s) => s.removeNode)
  const dragNodeId = useCanvasStore((s) => s.dragNodeId)
  const [hovered, setHovered] = useState(false)

  const { onPointerDown, onPointerMove, onPointerUp } = useNodeDrag(canvasRef, node.id)
  const isSelected = selectedNodeIds.has(node.id)
  const isDragging =
    dragNodeId === node.id ||
    (dragNodeId != null &&
      selectedNodeIds.has(dragNodeId) &&
      selectedNodeIds.has(node.id))

  const border = isSelected
    ? 'border-blue-500 ring-1 ring-blue-500/60'
    : isDark
      ? 'border-zinc-700'
      : 'border-zinc-200'

  const onHeaderMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    e.stopPropagation()
    if (e.ctrlKey || e.metaKey) toggleSelect(node.id)
    else if (!selectedNodeIds.has(node.id)) selectOnly(node.id)
  }

  const showInputPorts = node.type === 'gen-image-advanced' || node.type === 'gen-video'

  return (
    <div
      className={`absolute flex flex-col rounded-xl border shadow-lg select-none-canvas ${border} ${
        isDark ? 'bg-zinc-900' : 'bg-white'
      }`}
      style={{
        left: node.x,
        top: node.y,
        width: node.width,
        height: node.height,
        zIndex: isDragging ? 50 : 10,
        cursor: isDragging ? 'grabbing' : 'default',
      }}
      onMouseDown={onHeaderMouseDown}
      onDragStart={(e) => e.preventDefault()}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <OutputPort node={node} isDark={isDark} canvasRef={canvasRef} />
      {showInputPorts && (
        <InputPorts node={node} isDark={isDark} />
      )}
      <div
        className={`shrink-0 flex items-center gap-2 px-3 py-2 text-xs font-medium cursor-grab border-b rounded-t-xl overflow-hidden ${
          isDark ? 'bg-zinc-800/80 border-zinc-700 text-zinc-200' : 'bg-zinc-50 border-zinc-200 text-zinc-800'
        }`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <span className="truncate flex-1">{node.data.title || node.type}</span>
        <span className="opacity-50 text-[10px]">{node.type}</span>
        {hovered && (
          <button
            type="button"
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation()
              removeNode(node.id)
            }}
            className={`shrink-0 p-0.5 rounded transition-colors ${
              isDark ? 'hover:bg-red-900/60 text-zinc-500 hover:text-red-400' : 'hover:bg-red-100 text-zinc-400 hover:text-red-500'
            }`}
            title="Delete node"
          >
            <Trash2 className="size-3.5" />
          </button>
        )}
      </div>
      <div
        className={`flex-1 min-h-0 overflow-hidden rounded-b-xl ${isDark ? 'bg-zinc-900' : 'bg-white'}`}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <CanvasNodeBody node={node} isDark={isDark} />
      </div>
    </div>
  )
}

/* ── Output port (right side, draggable source) ── */

function OutputPort({
  node,
  isDark,
  canvasRef,
}: {
  node: NodeType
  isDark: boolean
  canvasRef: React.RefObject<HTMLElement | null>
}) {
  const addConnection = useCanvasStore((s) => s.addConnection)
  const view = useCanvasStore((s) => s.view)

  const dragRef = useRef<{ nodeId: string; x: number; y: number } | null>(null)
  const [dragging, setDragging] = useState<{ x: number; y: number } | null>(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  /* ── window-level move / up listeners (no pointer capture) ── */
  useEffect(() => {
    if (!dragging) return

    const onMove = (e: PointerEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (rect) {
        setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
      }
    }

    const onUp = (e: PointerEvent) => {
      if (!dragRef.current) return

      // Hit-test: find the closest [data-port="input"] element under the cursor
      const target = document.elementFromPoint(e.clientX, e.clientY)
      const portEl = target?.closest('[data-port="input"]') as HTMLElement | null
      if (portEl) {
        const toId = portEl.dataset.nodeId
        const portIdx = portEl.dataset.portIdx
        if (toId && toId !== dragRef.current.nodeId) {
          const idx = portIdx != null ? parseInt(portIdx, 10) : 0
          if (idx < 2) {
            // Check for existing connection on same port
            const conns = useCanvasStore.getState().connections
            const exists = conns.some(
              (c) => c.to === toId && c.toPort === `ref-${idx}`,
            )
            if (!exists) {
              addConnection({
                from: dragRef.current.nodeId,
                to: toId,
                toPort: `ref-${idx}`,
              })
            }
          }
        }
      }
      dragRef.current = null
      setDragging(null)
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [dragging, canvasRef, addConnection])

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.stopPropagation()
      e.preventDefault()
      const portX = node.x + node.width
      const portY = node.y + node.height / 2
      dragRef.current = { nodeId: node.id, x: portX, y: portY }
      setDragging({ x: portX, y: portY })
      const rect = canvasRef.current?.getBoundingClientRect()
      if (rect) {
        setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
      }
    },
    [node.id, node.x, node.y, node.width, node.height, canvasRef],
  )

  return (
    <>
      {dragging && createPortal(
        <svg
          className="pointer-events-none fixed inset-0 overflow-visible z-50"
          style={{ width: '100vw', height: '100vh' }}
        >
          {(() => {
            const rect = canvasRef.current?.getBoundingClientRect()
            if (!rect) return null
            const sx = dragging.x * view.zoom + view.x + rect.left
            const sy = dragging.y * view.zoom + view.y + rect.top
            const ex = mousePos.x + rect.left
            const ey = mousePos.y + rect.top
            const dist = Math.max(30, Math.abs(ex - sx) * 0.45)
            const d = `M ${sx} ${sy} C ${sx + dist} ${sy} ${ex - dist} ${ey} ${ex} ${ey}`
            return (
              <path
                d={d}
                fill="none"
                stroke="rgba(59,130,246,0.7)"
                strokeWidth={2}
                strokeDasharray="6 3"
              />
            )
          })()}
        </svg>,
        document.body,
      )}
      <div
        className="absolute right-0 w-3 h-3 rounded-full border-2 pointer-events-auto cursor-crosshair z-10 transition-colors hover:scale-125"
        style={{
          top: node.height / 2 - 6,
          transform: 'translateX(50%)',
          borderColor: isDark ? '#3b82f6' : '#2563eb',
          backgroundColor: isDark ? '#1e293b' : '#dbeafe',
        }}
        data-port="output"
        data-node-id={node.id}
        onPointerDown={handlePointerDown}
      />
    </>
  )
}

/* ── Input ports (left side, drop targets) ── */

function InputPorts({
  node,
  isDark,
}: {
  node: NodeType
  isDark: boolean
}) {
  const connections = useCanvasStore((s) => s.connections)

  const incoming = connections.filter((c) => c.to === node.id)
  const portPositions = [node.height * 0.3, node.height * 0.6]

  return (
    <>
      {portPositions.map((yPos, i) => {
        const conn = incoming.find((c) => c.toPort === `ref-${i}`) || incoming[i]
        const isConnected = !!conn
        return (
          <div
            key={i}
            className="absolute left-0 w-4 h-4 rounded-full border-2 pointer-events-auto z-10 transition-colors"
            style={{
              top: yPos - 8,
              transform: 'translateX(-50%)',
              borderColor: isDark ? '#3b82f6' : '#2563eb',
              backgroundColor: isConnected
                ? isDark ? '#3b82f6' : '#2563eb'
                : isDark ? '#1e293b' : '#dbeafe',
            }}
            data-port="input"
            data-node-id={node.id}
            data-port-idx={i}
          />
        )
      })}
    </>
  )
}
