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
  const dragNodeId = useCanvasStore((s) => s.dragNodeId)

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

  return (
    <div
      className={`absolute flex flex-col rounded-xl border shadow-lg select-none-canvas overflow-hidden ${border} ${
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
    >
      <div
        className={`shrink-0 flex items-center gap-2 px-3 py-2 text-xs font-medium cursor-grab border-b ${
          isDark ? 'bg-zinc-800/80 border-zinc-700 text-zinc-200' : 'bg-zinc-50 border-zinc-200 text-zinc-800'
        }`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <span className="truncate flex-1">{node.data.title || node.type}</span>
        <span className="opacity-50 text-[10px]">{node.type}</span>
      </div>
      <div
        className={`flex-1 min-h-0 overflow-hidden ${isDark ? 'bg-zinc-900' : 'bg-white'}`}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <CanvasNodeBody node={node} isDark={isDark} />
      </div>
    </div>
  )
}
