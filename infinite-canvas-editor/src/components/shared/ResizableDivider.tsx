import { useEffect, useRef, useState } from 'react'

type Props = {
  direction: 'horizontal' | 'vertical'
  /** Current size of the *first* pane (width if vertical drag, height if horizontal) */
  value: number
  onChange: (n: number) => void
  min?: number
  max?: number
  className?: string
  /** Flip drag direction (use when resizing a *bottom* pane — dragging up grows it) */
  invert?: boolean
}

/** Drag to resize: for horizontal divider, user drags vertically to change height of top/main area */
export function ResizableDivider({
  direction,
  value,
  onChange,
  min = 120,
  max = 560,
  className = '',
  invert = false,
}: Props) {
  const dragging = useRef(false)
  const start = useRef(0)
  const startVal = useRef(0)
  const [hover, setHover] = useState(false)

  useEffect(
    () => () => {
      dragging.current = false
      document.body.style.cursor = ''
    },
    [],
  )

  return (
    <div
      role="separator"
      aria-orientation={direction === 'horizontal' ? 'horizontal' : 'vertical'}
      onPointerDown={(e) => {
        e.preventDefault()
        dragging.current = true
        start.current = direction === 'horizontal' ? e.clientY : e.clientX
        startVal.current = value
        document.body.style.cursor =
          direction === 'horizontal' ? 'row-resize' : 'col-resize'

        const onMove = (ev: PointerEvent) => {
          if (!dragging.current) return
          const delta =
            direction === 'horizontal' ? ev.clientY - start.current : ev.clientX - start.current
          const next = Math.min(max, Math.max(min, startVal.current + (invert ? -delta : delta)))
          onChange(next)
        }
        const onUp = () => {
          dragging.current = false
          document.body.style.cursor = ''
          window.removeEventListener('pointermove', onMove)
          window.removeEventListener('pointerup', onUp)
        }
        window.addEventListener('pointermove', onMove)
        window.addEventListener('pointerup', onUp)
      }}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      className={`shrink-0 bg-zinc-800 transition-colors ${
        direction === 'horizontal' ? 'h-2 w-full cursor-row-resize' : 'w-2 h-full cursor-col-resize'
      } ${hover ? 'bg-blue-600/60' : ''} ${className}`}
    />
  )
}
