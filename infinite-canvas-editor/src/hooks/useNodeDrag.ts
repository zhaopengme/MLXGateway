import { useCallback, useRef } from 'react'
import { useCanvasStore } from '../stores/canvasStore'
import { screenToWorld } from '../utils/coordinates'

export function useNodeDrag(
  canvasRef: React.RefObject<HTMLElement | null>,
  nodeId: string,
) {
  const view = useCanvasStore((s) => s.view)
  const nodes = useCanvasStore((s) => s.nodes)
  const updateNode = useCanvasStore((s) => s.updateNode)
  const setDragNodeId = useCanvasStore((s) => s.setDragNodeId)
  const selectedNodeIds = useCanvasStore((s) => s.selectedNodeIds)

  const dragRef = useRef<{
    startX: number
    startY: number
    nodeStarts: Map<string, { x: number; y: number }>
  } | null>(null)

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button !== 0) return
      const target = e.target as HTMLElement
      if (target.closest('button, a, input, textarea, video')) return

      e.stopPropagation()
      e.preventDefault()

      const rect = canvasRef.current?.getBoundingClientRect()
      const w = screenToWorld(e.clientX, e.clientY, view, rect)
      const starts = new Map<string, { x: number; y: number }>()
      const toMove =
        selectedNodeIds.has(nodeId) && selectedNodeIds.size > 0
          ? [...selectedNodeIds]
          : [nodeId]

      for (const id of toMove) {
        const n = nodes.find((x) => x.id === id)
        if (n) starts.set(id, { x: n.x, y: n.y })
      }

      dragRef.current = { startX: w.x, startY: w.y, nodeStarts: starts }
      setDragNodeId(nodeId)
      ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    },
    [canvasRef, nodeId, nodes, selectedNodeIds, setDragNodeId, view],
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current) return
      const rect = canvasRef.current?.getBoundingClientRect()
      const w = screenToWorld(e.clientX, e.clientY, view, rect)
      const dx = w.x - dragRef.current.startX
      const dy = w.y - dragRef.current.startY
      for (const [id, start] of dragRef.current.nodeStarts) {
        updateNode(id, { x: start.x + dx, y: start.y + dy })
      }
    },
    [canvasRef, updateNode, view],
  )

  const onPointerUp = useCallback(
    (e: React.PointerEvent) => {
      dragRef.current = null
      setDragNodeId(null)
      try {
        ;(e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId)
      } catch {
        /* ok */
      }
    },
    [setDragNodeId],
  )

  return { onPointerDown, onPointerMove, onPointerUp }
}
