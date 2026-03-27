import type { ViewState } from '../types/canvas'

export function screenToWorld(
  sx: number,
  sy: number,
  view: ViewState,
  rect: DOMRect | undefined,
): { x: number; y: number } {
  const localX = rect ? sx - rect.left : sx
  const localY = rect ? sy - rect.top : sy
  return {
    x: (localX - view.x) / view.zoom,
    y: (localY - view.y) / view.zoom,
  }
}

export function worldToScreen(
  wx: number,
  wy: number,
  view: ViewState,
  rect: DOMRect | undefined,
): { x: number; y: number } {
  const sx = wx * view.zoom + view.x
  const sy = wy * view.zoom + view.y
  if (!rect) return { x: sx, y: sy }
  return { x: sx + rect.left, y: sy + rect.top }
}
