import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { Trash2 } from 'lucide-react'
import type { TimelineClip } from '../../types/timeline'
import { useTimelineStore } from '../../stores/timelineStore'

const MIN_CLIP = 0.15
const MIN_IMAGE = 0.5

type Props = {
  trackId: string
  clip: TimelineClip
  pxPerSec: number
  isDark: boolean
  isSelected: boolean
  onSelect: () => void
}

function readLiveClip(trackId: string, clipId: string): TimelineClip | null {
  const t = useTimelineStore.getState().tracks.find((x) => x.id === trackId)
  return t?.clips.find((c) => c.id === clipId) ?? null
}

export function TimelineClipItem({
  trackId,
  clip,
  pxPerSec,
  isDark,
  isSelected,
  onSelect,
}: Props) {
  const removeClip = useTimelineStore((s) => s.removeClip)
  const updateClip = useTimelineStore((s) => s.updateClip)

  const effective =
    clip.mediaType === 'image'
      ? Math.max(MIN_IMAGE, clip.duration)
      : Math.max(MIN_CLIP, clip.duration - clip.trimStart - clip.trimEnd)

  const w = Math.max(48, effective * pxPerSec)

  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: clip.id,
  })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    width: w,
  }

  const ring = isSelected ? 'ring-2 ring-blue-500 ring-offset-1 ring-offset-transparent' : ''

  return (
    <div
      ref={setNodeRef}
      data-timeline-clip
      style={style}
      className={`relative shrink-0 h-14 rounded-lg border overflow-hidden group ${
        isDragging ? 'opacity-70 z-10' : ''
      } ${isDark ? 'border-zinc-600 bg-zinc-800' : 'border-zinc-300 bg-white'} ${ring}`}
    >
      <button
        type="button"
        className="absolute top-1 right-1 z-10 p-1 rounded opacity-0 group-hover:opacity-100 bg-red-600/90 text-white"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation()
          removeClip(trackId, clip.id)
        }}
      >
        <Trash2 className="size-3" />
      </button>
      <div className="flex h-full">
        {clip.mediaType !== 'image' ? (
          <div
            className={`w-2 shrink-0 cursor-ew-resize ${isDark ? 'bg-zinc-600' : 'bg-zinc-300'}`}
            onPointerDown={(e) => {
              e.stopPropagation()
              const startX = e.clientX
              const startTrim = clip.trimStart
              const onMove = (ev: PointerEvent) => {
                const cur = readLiveClip(trackId, clip.id)
                if (!cur || cur.mediaType === 'image') return
                const deltaSec = (ev.clientX - startX) / pxPerSec
                const maxTrimStart = Math.max(0, cur.duration - cur.trimEnd - MIN_CLIP)
                const next = Math.max(0, Math.min(maxTrimStart, startTrim + deltaSec))
                updateClip(trackId, clip.id, { trimStart: next })
              }
              const onUp = () => {
                window.removeEventListener('pointermove', onMove)
                window.removeEventListener('pointerup', onUp)
              }
              window.addEventListener('pointermove', onMove)
              window.addEventListener('pointerup', onUp)
            }}
          />
        ) : (
          <div className="w-0 shrink-0 overflow-hidden" aria-hidden />
        )}
        <div
          className="flex-1 flex items-stretch min-w-0 cursor-grab active:cursor-grabbing"
          {...attributes}
          {...listeners}
          onPointerDown={(e) => {
            if (e.button === 0) onSelect()
          }}
        >
          {clip.mediaType === 'audio' ? (
            <div
              className={`h-full w-16 shrink-0 bg-gradient-to-br from-amber-900/50 via-violet-900/40 to-sky-900/50 ${
                isDark ? '' : 'from-amber-200/80 via-violet-200/80 to-sky-200/80'
              }`}
              title="Audio"
            />
          ) : clip.thumbnailDataUrl ? (
            <img src={clip.thumbnailDataUrl} alt="" className="h-full w-16 object-cover shrink-0" />
          ) : (
            <div className={`h-full w-16 shrink-0 ${isDark ? 'bg-zinc-700' : 'bg-zinc-200'}`} />
          )}
          <div className="px-2 py-1 min-w-0 flex flex-col justify-center">
            <span
              className={`text-[10px] font-medium truncate ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}
            >
              {clip.label || (clip.mediaType === 'video' ? 'Video' : clip.mediaType === 'audio' ? 'Audio' : 'Image')}
            </span>
            <span className={`text-[9px] ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
              {clip.mediaType === 'image'
                ? `${clip.duration.toFixed(1)}s`
                : `${effective.toFixed(1)}s`}
            </span>
          </div>
        </div>
        <div
          className={`w-2 shrink-0 cursor-ew-resize ${isDark ? 'bg-zinc-600' : 'bg-zinc-300'}`}
          onPointerDown={(e) => {
            e.stopPropagation()
            const startX = e.clientX
            const startTrimEnd = clip.trimEnd
            const startDur = clip.duration
            const onMove = (ev: PointerEvent) => {
              const cur = readLiveClip(trackId, clip.id)
              if (!cur) return
              const deltaSec = (startX - ev.clientX) / pxPerSec
              if (cur.mediaType === 'image') {
                const nextDur = Math.max(MIN_IMAGE, startDur + deltaSec)
                updateClip(trackId, clip.id, { duration: nextDur })
                return
              }
              const maxTrimEnd = Math.max(0, cur.duration - cur.trimStart - MIN_CLIP)
              const next = Math.max(0, Math.min(maxTrimEnd, startTrimEnd + deltaSec))
              updateClip(trackId, clip.id, { trimEnd: next })
            }
            const onUp = () => {
              window.removeEventListener('pointermove', onMove)
              window.removeEventListener('pointerup', onUp)
            }
            window.addEventListener('pointermove', onMove)
            window.addEventListener('pointerup', onUp)
          }}
        />
      </div>
    </div>
  )
}
