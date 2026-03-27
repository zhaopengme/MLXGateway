import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import { SortableContext, horizontalListSortingStrategy } from '@dnd-kit/sortable'
import { Film, Pause, Play, Plus, Scissors, Download } from 'lucide-react'
import { useCallback, useEffect, useRef } from 'react'
import {
  useTimelineStore,
  totalDurationSec,
  effectiveClipDuration,
} from '../../stores/timelineStore'
import { useCanvasStore, guessMediaTypeFromNode } from '../../stores/canvasStore'
import { TimelineClipItem } from './TimelineClipItem'
import { useVideoExport } from '../../hooks/useVideoExport'
import { TimeRuler } from './TimeRuler'
import { MAIN_TRACK_ID, AUDIO_TRACK_ID } from '../../types/timeline'
import { getAudioDurationSec, getVideoDurationSec } from '../../utils/mediaDuration'

type Props = {
  isDark: boolean
  timelineHeight: number
}

export function TimelinePanel({ isDark, timelineHeight }: Props) {
  const tracks = useTimelineStore((s) => s.tracks)
  const pxPerSec = useTimelineStore((s) => s.pxPerSec)
  const playheadSec = useTimelineStore((s) => s.playheadSec)
  const isPlaying = useTimelineStore((s) => s.isPlaying)
  const selectedClipId = useTimelineStore((s) => s.selectedClipId)
  const selectClip = useTimelineStore((s) => s.selectClip)
  const setPlayheadSec = useTimelineStore((s) => s.setPlayheadSec)
  const setPxPerSec = useTimelineStore((s) => s.setPxPerSec)
  const setPlaying = useTimelineStore((s) => s.setPlaying)
  const reorderClips = useTimelineStore((s) => s.reorderClips)
  const addClip = useTimelineStore((s) => s.addClip)
  const insertClipAfter = useTimelineStore((s) => s.insertClipAfter)
  const updateClip = useTimelineStore((s) => s.updateClip)

  const nodes = useCanvasStore((s) => s.nodes)
  const clearNodeSelectionOnly = useCanvasStore((s) => s.clearNodeSelectionOnly)
  const { exportMp4, loading: exportLoading, progress, label } = useVideoExport()

  const trackRef = useRef<HTMLDivElement>(null)
  const rafRef = useRef<number>(0)
  const lastTRef = useRef<number>(0)

  const mainClips = tracks.find((t) => t.id === MAIN_TRACK_ID)?.clips ?? []
  const totalDur = totalDurationSec(tracks)

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      return
    }
    lastTRef.current = performance.now()
    const tick = (now: number) => {
      const dt = (now - lastTRef.current) / 1000
      lastTRef.current = now
      const st = useTimelineStore.getState()
      const t = st.playheadSec + dt
      if (t >= totalDurationSec(st.tracks)) {
        st.setPlaying(false)
        st.setPlayheadSec(0)
        return
      }
      st.setPlayheadSec(t)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [isPlaying, tracks])

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }))

  const onDragEnd = (e: DragEndEvent) => {
    const { active, over } = e
    if (!over || active.id === over.id) return
    const st = useTimelineStore.getState()
    let activeTrack: string | null = null
    let overTrack: string | null = null
    for (const t of st.tracks) {
      if (t.clips.some((c) => c.id === active.id)) activeTrack = t.id
      if (t.clips.some((c) => c.id === over.id)) overTrack = t.id
    }
    if (!activeTrack || !overTrack || activeTrack !== overTrack) return
    const clips = st.tracks.find((t) => t.id === activeTrack)!.clips
    const ids = clips.map((c) => c.id)
    const oi = ids.indexOf(over.id as string)
    const ai = ids.indexOf(active.id as string)
    if (oi < 0 || ai < 0) return
    const next = [...ids]
    next.splice(ai, 1)
    next.splice(oi, 0, active.id as string)
    reorderClips(activeTrack, next)
  }

  const onDropTimeline = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault()
      const raw = e.dataTransfer.getData('application/x-canvas-node')
      if (raw) {
        const node = nodes.find((n) => n.id === raw)
        if (!node) return
        const mt = guessMediaTypeFromNode(node)
        const url =
          mt === 'audio' ? node.data.audioBlobUrl : node.data.content
        if (!mt || !url) return
        const trackId = mt === 'audio' ? AUDIO_TRACK_ID : MAIN_TRACK_ID
        let duration: number
        if (mt === 'image') duration = 3
        else if (mt === 'audio') duration = await getAudioDurationSec(url)
        else duration = await getVideoDurationSec(url)
        await addClip({
          mediaType: mt,
          src: url,
          duration,
          trimStart: 0,
          trimEnd: 0,
          sourceNodeId: node.id,
          label: node.data.title || node.type,
          trackId,
        })
        return
      }
      const hist = e.dataTransfer.getData('application/x-history-item')
      if (hist) {
        try {
          const item = JSON.parse(hist) as { type: 'image' | 'video'; url: string; prompt?: string }
          const dur =
            item.type === 'image' ? 3 : await getVideoDurationSec(item.url)
          await addClip({
            mediaType: item.type,
            src: item.url,
            duration: dur,
            trimStart: 0,
            trimEnd: 0,
            label: item.prompt?.slice(0, 24) || item.type,
            trackId: MAIN_TRACK_ID,
          })
        } catch {
          /* ignore */
        }
      }
    },
    [addClip, nodes],
  )

  const splitAtPlayhead = () => {
    if (!mainClips.length) return
    let acc = 0
    const t = playheadSec
    for (let i = 0; i < mainClips.length; i++) {
      const c = mainClips[i]
      const len = effectiveClipDuration(c)
      if (t < acc + len) {
        if (c.mediaType !== 'video') return
        const p = t - acc
        if (p < 0.15 || p > len - 0.15) return
        const L = c.duration - c.trimStart - c.trimEnd
        updateClip(MAIN_TRACK_ID, c.id, { trimEnd: c.trimEnd + (L - p) })
        void insertClipAfter(MAIN_TRACK_ID, c.id, {
          mediaType: 'video',
          src: c.src,
          duration: c.duration,
          trimStart: c.trimStart + p,
          trimEnd: c.trimEnd,
          label: `${c.label || 'clip'} (2)`,
          thumbnailDataUrl: c.thumbnailDataUrl,
        })
        return
      }
      acc += len
    }
  }

  const playheadPx = playheadSec * pxPerSec
  const contentW = Math.max(400, totalDur * pxPerSec + 80)

  const bg = isDark ? 'bg-zinc-900 border-zinc-800' : 'bg-zinc-50 border-zinc-200'

  return (
    <div className={`flex flex-col border-t shrink-0 ${bg}`} style={{ height: timelineHeight }}>
      <div
        className={`flex items-center gap-2 px-3 py-2 border-b ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`}
      >
        <Film className={`size-4 ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`} />
        <span className={`text-xs font-semibold ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
          Timeline
        </span>
        <span className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
          {tracks.reduce((n, t) => n + t.clips.length, 0)} clips · {totalDur.toFixed(1)}s
        </span>
        <div className="flex-1" />
        {exportLoading && (
          <span className="text-[10px] text-blue-400">
            {label} {progress}%
          </span>
        )}
        <button
          type="button"
          onClick={() => setPlaying(!isPlaying)}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
            isDark ? 'bg-zinc-800 text-zinc-200' : 'bg-white border border-zinc-200'
          }`}
        >
          {isPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          onClick={splitAtPlayhead}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
            isDark ? 'bg-zinc-800 text-zinc-200' : 'bg-white border border-zinc-200'
          }`}
        >
          <Scissors className="size-3.5" />
          Split
        </button>
        <label
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer ${
            isDark ? 'bg-zinc-800 text-zinc-200' : 'bg-white border border-zinc-200'
          }`}
        >
          <Plus className="size-3.5" />
          Import
          <input
            type="file"
            accept="video/*,image/*,audio/*"
            multiple
            className="hidden"
            onChange={async (ev) => {
              const files = ev.target.files
              if (!files) return
              for (const f of files) {
                const url = URL.createObjectURL(f)
                const isVid = f.type.startsWith('video')
                const isAud = f.type.startsWith('audio')
                let duration: number
                if (isAud) duration = await getAudioDurationSec(url)
                else if (isVid) duration = await getVideoDurationSec(url)
                else duration = 3
                await addClip({
                  mediaType: isAud ? 'audio' : isVid ? 'video' : 'image',
                  src: url,
                  duration,
                  trimStart: 0,
                  trimEnd: 0,
                  label: f.name,
                  trackId: isAud ? AUDIO_TRACK_ID : MAIN_TRACK_ID,
                })
              }
              ev.target.value = ''
            }}
          />
        </label>
        <button
          type="button"
          disabled={exportLoading || !mainClips.length}
          onClick={() => void exportMp4()}
          className={`flex items-center gap-1 px-2 py-1 rounded text-xs disabled:opacity-40 ${
            isDark ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
          }`}
        >
          <Download className="size-3.5" />
          Export MP4
        </button>
        <span className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>Zoom</span>
        <input
          type="range"
          min={8}
          max={200}
          value={pxPerSec}
          onChange={(e) => setPxPerSec(Number(e.target.value))}
          className="w-24"
        />
      </div>
      <div
        ref={trackRef}
        className={`flex-1 min-h-0 overflow-x-auto overflow-y-auto relative px-0 py-0 ${
          isDark ? 'bg-zinc-950/50' : 'bg-white/50'
        }`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDropTimeline}
      >
        <div className="flex">
          <div
            className={`w-14 shrink-0 border-r flex flex-col ${isDark ? 'border-zinc-800 bg-zinc-900/60' : 'border-zinc-200 bg-zinc-100/80'}`}
          >
            <div className={`h-6 shrink-0 border-b ${isDark ? 'border-zinc-800' : 'border-zinc-200'}`} />
            {tracks.map((tr) => (
              <div
                key={tr.id}
                className={`h-14 shrink-0 flex items-center justify-center text-[9px] font-medium px-1 text-center border-b ${
                  isDark ? 'border-zinc-800 text-zinc-400' : 'border-zinc-200 text-zinc-600'
                }`}
              >
                {tr.label}
              </div>
            ))}
          </div>
          <div className="flex-1 min-w-0 overflow-x-visible">
            <div
              className="relative"
              style={{ width: contentW }}
              onPointerDown={(e) => {
                const t = e.target as HTMLElement
                if (t.closest('[data-timeline-clip]')) return
                if (t.closest('[data-timeline-playhead]')) return
                selectClip(null)
              }}
            >
              <TimeRuler totalSec={totalDur} pxPerSec={pxPerSec} isDark={isDark} />
              <div
                className="relative"
                style={{ minHeight: tracks.length * 56 }}
              >
                <div
                  className="absolute top-0 bottom-0 w-px z-20 bg-red-500 pointer-events-none"
                  style={{ left: playheadPx }}
                />
                <button
                  type="button"
                  data-timeline-playhead
                  className="absolute top-0 bottom-0 w-3 -ml-1.5 z-30 cursor-ew-resize bg-red-500/30 hover:bg-red-500/50"
                  style={{ left: playheadPx }}
                  onPointerDown={(e) => {
                    e.preventDefault()
                    const track = trackRef.current
                    if (!track) return
                    const rect = track.getBoundingClientRect()
                    const labelW = 56
                    const scrollLeft = track.scrollLeft
                    const onMove = (ev: PointerEvent) => {
                      const x = ev.clientX - rect.left + scrollLeft - labelW
                      setPlayheadSec(Math.max(0, Math.min(totalDur, x / pxPerSec)))
                    }
                    const onUp = () => {
                      window.removeEventListener('pointermove', onMove)
                      window.removeEventListener('pointerup', onUp)
                    }
                    window.addEventListener('pointermove', onMove)
                    window.addEventListener('pointerup', onUp)
                  }}
                />
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
                  {tracks.map((tr) => (
                    <div
                      key={tr.id}
                      className={`flex gap-1 items-center h-14 pt-1 border-b ${isDark ? 'border-zinc-800/80' : 'border-zinc-200/80'}`}
                      style={{ paddingLeft: 4 }}
                    >
                      <SortableContext items={tr.clips.map((c) => c.id)} strategy={horizontalListSortingStrategy}>
                        {tr.clips.map((c) => (
                          <TimelineClipItem
                            key={c.id}
                            trackId={tr.id}
                            clip={c}
                            pxPerSec={pxPerSec}
                            isDark={isDark}
                            isSelected={selectedClipId === c.id}
                            onSelect={() => {
                              clearNodeSelectionOnly()
                              selectClip(c.id)
                            }}
                          />
                        ))}
                      </SortableContext>
                    </div>
                  ))}
                </DndContext>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
