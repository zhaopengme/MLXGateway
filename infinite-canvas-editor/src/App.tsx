import { Moon, Server, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Toaster } from 'sonner'
import { ImageEditModal } from './components/ImageEditModal'
import { InfiniteCanvas } from './components/canvas/InfiniteCanvas'
import { PreviewMonitor } from './components/timeline/PreviewMonitor'
import { PropertiesPanel } from './components/properties/PropertiesPanel'
import { ResizableDivider } from './components/shared/ResizableDivider'
import { GatewaySettingsPanel } from './components/settings/GatewaySettingsPanel'
import { Sidebar } from './components/sidebar/Sidebar'
import { TimelinePanel } from './components/timeline/TimelinePanel'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useGatewayStore } from './stores/gatewayStore'
import { useProjectStore } from './stores/projectStore'

const TIMELINE_H_KEY = 'infinite-canvas-timeline-h'

export default function App() {
  useKeyboardShortcuts()
  const theme = useProjectStore((s) => s.theme)
  const setTheme = useProjectStore((s) => s.setTheme)
  const connected = useGatewayStore((s) => s.connected)
  const checkHealth = useGatewayStore((s) => s.checkHealth)
  const isDark = theme === 'dark'

  const [timelineH, setTimelineH] = useState(() => {
    const n = Number(localStorage.getItem(TIMELINE_H_KEY))
    return Number.isFinite(n) ? Math.min(420, Math.max(140, n)) : 220
  })
  const [gwOpen, setGwOpen] = useState(false)

  useEffect(() => {
    localStorage.setItem(TIMELINE_H_KEY, String(timelineH))
  }, [timelineH])

  useEffect(() => {
    void checkHealth()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot on mount
  }, [])

  return (
    <div
      className={`h-svh w-full flex flex-col overflow-hidden ${
        isDark ? 'bg-zinc-950 text-zinc-100' : 'bg-zinc-100 text-zinc-900'
      }`}
    >
      <header
        className={`shrink-0 flex items-center gap-3 px-4 py-2 border-b ${
          isDark ? 'border-zinc-800 bg-zinc-900' : 'border-zinc-200 bg-white'
        }`}
      >
        <h1 className="text-sm font-semibold tracking-tight">Infinite Canvas · MLXGateway</h1>
        <span className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
          Wheel zoom · Middle-click / Pan · Drag media to timeline
        </span>
        <div className="flex-1" />
        <span
          className={`text-[10px] font-medium hidden sm:inline ${connected ? 'text-green-500' : 'text-zinc-500'}`}
        >
          API {connected ? 'online' : 'offline'}
        </span>
        <button
          type="button"
          onClick={() => setGwOpen(true)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${
            isDark ? 'bg-zinc-800 text-zinc-200' : 'bg-zinc-100 text-zinc-800'
          }`}
        >
          <Server className="size-4" />
          Gateway
        </button>
        <button
          type="button"
          onClick={() => setTheme(isDark ? 'light' : 'dark')}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs ${
            isDark ? 'bg-zinc-800 text-zinc-200' : 'bg-zinc-100 text-zinc-800'
          }`}
        >
          {isDark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          {isDark ? 'Light' : 'Dark'}
        </button>
      </header>

      {gwOpen && (
        <>
          <button
            type="button"
            aria-label="Close gateway settings"
            className="fixed inset-0 z-40 bg-black/40"
            onClick={() => setGwOpen(false)}
          />
          <GatewaySettingsPanel isDark={isDark} open={gwOpen} onClose={() => setGwOpen(false)} />
        </>
      )}

      <ImageEditModal isDark={isDark} />

      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex-1 flex min-h-0">
          <Sidebar isDark={isDark} />
          <div className="flex-1 flex flex-col min-h-0 min-w-0">
            <PreviewMonitor isDark={isDark} />
            <InfiniteCanvas isDark={isDark} />
          </div>
          <PropertiesPanel isDark={isDark} />
        </div>
        <ResizableDivider
          direction="horizontal"
          value={timelineH}
          onChange={setTimelineH}
          min={120}
          max={480}
          invert
        />
        <TimelinePanel isDark={isDark} timelineHeight={timelineH} />
      </div>
      <Toaster
        position="top-center"
        expand={false}
        richColors
        closeButton
      />
    </div>
  )
}
