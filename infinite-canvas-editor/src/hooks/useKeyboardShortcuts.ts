import { useEffect } from 'react'
import { useCanvasStore } from '../stores/canvasStore'

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().includes('MAC')
      const mod = isMac ? e.metaKey : e.ctrlKey

      if (mod && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        useCanvasStore.temporal.getState().undo()
        return
      }
      if (mod && e.key === 'Z') {
        e.preventDefault()
        useCanvasStore.temporal.getState().redo()
        return
      }
      if (mod && e.key === 'y') {
        e.preventDefault()
        useCanvasStore.temporal.getState().redo()
        return
      }

      if (e.key === 'Delete' || e.key === 'Backspace') {
        const tag = (e.target as HTMLElement).tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA') return
        const selected = useCanvasStore.getState().selectedNodeIds
        if (selected.size > 0) {
          e.preventDefault()
          const store = useCanvasStore.getState()
          for (const id of [...selected]) {
            store.removeNode(id)
          }
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
}
