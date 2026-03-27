import { useEffect, useMemo, useRef } from 'react'
import { useTimelineStore, getMainTrackClipAtTime } from '../../stores/timelineStore'

type Props = { isDark: boolean }

export function PreviewMonitor({ isDark }: Props) {
  const tracks = useTimelineStore((s) => s.tracks)
  const playheadSec = useTimelineStore((s) => s.playheadSec)
  const isPlaying = useTimelineStore((s) => s.isPlaying)

  const hit = useMemo(
    () => getMainTrackClipAtTime(tracks, playheadSec),
    [tracks, playheadSec],
  )

  const videoRef = useRef<HTMLVideoElement>(null)
  const clipId = hit?.clip.id
  const clipType = hit?.clip.mediaType
  const localT = hit?.localT ?? 0

  useEffect(() => {
    const v = videoRef.current
    if (!v || clipType !== 'video' || !clipId) return
    const t = Math.max(0, localT)
    if (Number.isFinite(t) && Math.abs(v.currentTime - t) > 0.04) {
      try {
        v.currentTime = t
      } catch {
        /* seek while loading */
      }
    }
  }, [clipId, clipType, localT, playheadSec])

  useEffect(() => {
    const v = videoRef.current
    if (!v || clipType !== 'video' || !clipId) return
    if (isPlaying) {
      void v.play().catch(() => {})
    } else {
      v.pause()
    }
  }, [isPlaying, clipType, clipId])

  const wrap = `flex flex-col shrink-0 border-b ${
    isDark ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-zinc-50/80'
  }`

  if (!hit) {
    return (
      <div className={`${wrap} px-3 py-2`}>
        <span className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>
          Preview · move playhead over a main-track clip
        </span>
      </div>
    )
  }

  const { clip } = hit

  if (clip.mediaType === 'audio') {
    return (
      <div className={`${wrap} px-3 py-2 gap-1`}>
        <span className={`text-[10px] ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
          Main track: audio-only clip — use the Audio track row; preview shows video/image only.
        </span>
        <audio src={clip.src} controls className="w-full max-w-md" />
      </div>
    )
  }

  return (
    <div className={`${wrap} px-3 py-2 gap-1`}>
      <span className={`text-[10px] font-medium ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
        Preview — {clip.label || clip.mediaType}{' '}
        <span className="opacity-60">@ {playheadSec.toFixed(2)}s</span>
      </span>
      <div className="relative rounded-lg overflow-hidden flex items-center justify-center max-h-[200px] min-h-[120px] bg-black">
        {clip.mediaType === 'image' ? (
          <img src={clip.src} alt="" className="max-h-[200px] w-full object-contain" />
        ) : (
          <video
            key={clip.id}
            ref={videoRef}
            src={clip.src}
            className="max-h-[200px] w-full object-contain"
            playsInline
            controls
            muted={false}
          />
        )}
      </div>
    </div>
  )
}
