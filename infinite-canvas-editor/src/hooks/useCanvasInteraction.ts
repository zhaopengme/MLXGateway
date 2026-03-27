import { useCallback, useEffect, useRef } from 'react'
import { useCanvasStore } from '../stores/canvasStore'
import { screenToWorld } from '../utils/coordinates'

const MIN_ZOOM = 0.15
const MAX_ZOOM = 4

export function useCanvasInteraction(canvasRef: React.RefObject<HTMLElement | null>) {
  const view = useCanvasStore((s) => s.view)
  const setView = useCanvasStore((s) => s.setView)
  const tool = useCanvasStore((s) => s.tool)
  const clearSelection = useCanvasStore((s) => s.clearSelection)

  const panningRef = useRef(false)
  const panStartRef = useRef({ x: 0, y: 0, vx: 0, vy: 0 })

  useEffect(() => {
    const el = canvasRef.current
    if (!el) return

    const wheelHandler = (e: WheelEvent) => {
      if (e.ctrlKey) {
        e.preventDefault()
        return
      }
      try {
        if (e.cancelable) e.preventDefault()
      } catch {
        /* passive */
      }

      const rect = el.getBoundingClientRect()
      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top
      setView((prev) => {
        const zoomFactor = e.deltaY > 0 ? 0.92 : 1.08
        let newZoom = Math.min(Math.max(prev.zoom * zoomFactor, MIN_ZOOM), MAX_ZOOM)
        newZoom = Math.round(newZoom * 1000) / 1000
        const scale = newZoom / prev.zoom
        const newX = mouseX - (mouseX - prev.x) * scale
        const newY = mouseY - (mouseY - prev.y) * scale
        return { zoom: newZoom, x: newX, y: newY }
      })
    }

    el.addEventListener('wheel', wheelHandler, { passive: false })
    return () => el.removeEventListener('wheel', wheelHandler)
  }, [canvasRef, setView])

  const onCanvasPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button === 1 || (tool === 'pan' && e.button === 0)) {
        e.preventDefault()
        panningRef.current = true
        panStartRef.current = {
          x: e.clientX,
          y: e.clientY,
          vx: view.x,
          vy: view.y,
        }
        ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
        return
      }
      if (e.button === 0 && e.target === e.currentTarget) {
        clearSelection()
      }
    },
    [tool, view.x, view.y, clearSelection],
  )

  const onCanvasPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!panningRef.current) return
      const dx = e.clientX - panStartRef.current.x
      const dy = e.clientY - panStartRef.current.y
      setView((v) => ({
        ...v,
        x: panStartRef.current.vx + dx,
        y: panStartRef.current.vy + dy,
      }))
    },
    [setView],
  )

  const onCanvasPointerUp = useCallback((e: React.PointerEvent) => {
    panningRef.current = false
    try {
      ;(e.target as HTMLElement).releasePointerCapture?.(e.pointerId)
    } catch {
      /* ok */
    }
  }, [])

  return {
    onCanvasPointerDown,
    onCanvasPointerMove,
    onCanvasPointerUp,
    screenToWorld: (sx: number, sy: number) =>
      screenToWorld(sx, sy, view, canvasRef.current?.getBoundingClientRect()),
  }
}
