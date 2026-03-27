import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { TimelineClip, TimelineTrack } from '../types/timeline'
import {
  AUDIO_TRACK_ID,
  MAIN_TRACK_ID,
} from '../types/timeline'
import { imageScaledDataUrl, videoFirstFrameDataUrl } from '../utils/thumbnails'

function newId() {
  return crypto.randomUUID()
}

export function effectiveClipDuration(c: TimelineClip): number {
  if (c.mediaType === 'image') return Math.max(0.5, c.duration)
  return Math.max(0.1, c.duration - c.trimStart - c.trimEnd)
}

function trackDurationSec(track: TimelineTrack): number {
  return track.clips.reduce((sum, c) => sum + effectiveClipDuration(c), 0)
}

/** Project length: longest track (parallel audio + video). */
export function totalDurationSec(tracks: TimelineTrack[]): number {
  if (!tracks.length) return 0
  return Math.max(0, ...tracks.map(trackDurationSec))
}

function createDefaultTracks(): TimelineTrack[] {
  return [
    { id: MAIN_TRACK_ID, kind: 'main', label: 'Video', clips: [] },
    { id: AUDIO_TRACK_ID, kind: 'audio', label: 'Audio', clips: [] },
  ]
}

function stripBlobClips(tracks: TimelineTrack[]): TimelineTrack[] {
  return tracks.map((t) => ({
    ...t,
    clips: t.clips.filter((c) => !c.src.startsWith('blob:')),
  }))
}

async function fillThumbnail(clip: TimelineClip): Promise<TimelineClip> {
  if (clip.thumbnailDataUrl || clip.mediaType === 'audio') return clip
  try {
    if (clip.mediaType === 'video') {
      const th = await videoFirstFrameDataUrl(clip.src)
      return { ...clip, thumbnailDataUrl: th }
    }
    const th = await imageScaledDataUrl(clip.src)
    return { ...clip, thumbnailDataUrl: th }
  } catch {
    return clip
  }
}

export type AddClipInput = Omit<TimelineClip, 'id'> & {
  id?: string
  trackId?: string
}

interface TimelineState {
  tracks: TimelineTrack[]
  selectedClipId: string | null
  playheadSec: number
  pxPerSec: number
  isPlaying: boolean

  selectClip: (id: string | null) => void
  findClip: (clipId: string) => { trackId: string; clip: TimelineClip } | null
  getMainTrackClips: () => TimelineClip[]
  getAudioTrackClips: () => TimelineClip[]

  addClip: (partial: AddClipInput) => Promise<string>
  insertClipAfter: (
    trackId: string,
    afterClipId: string,
    partial: Omit<TimelineClip, 'id'> & { id?: string },
  ) => Promise<string>
  removeClip: (trackId: string, clipId: string) => void
  reorderClips: (trackId: string, orderedIds: string[]) => void
  updateClip: (trackId: string, clipId: string, patch: Partial<TimelineClip>) => void
  setPlayheadSec: (t: number) => void
  setPxPerSec: (v: number) => void
  setPlaying: (v: boolean) => void
  clearClips: () => void
}

function patchTracks(
  tracks: TimelineTrack[],
  trackId: string,
  fn: (clips: TimelineClip[]) => TimelineClip[],
): TimelineTrack[] {
  return tracks.map((t) => (t.id === trackId ? { ...t, clips: fn(t.clips) } : t))
}

export const useTimelineStore = create<TimelineState>()(
  persist(
    (set, get) => ({
      tracks: createDefaultTracks(),
      selectedClipId: null,
      playheadSec: 0,
      pxPerSec: 40,
      isPlaying: false,

      selectClip: (id) => set({ selectedClipId: id }),

      findClip: (clipId) => {
        for (const t of get().tracks) {
          const clip = t.clips.find((c) => c.id === clipId)
          if (clip) return { trackId: t.id, clip }
        }
        return null
      },

      getMainTrackClips: () =>
        get().tracks.find((t) => t.id === MAIN_TRACK_ID)?.clips ?? [],

      getAudioTrackClips: () =>
        get().tracks.find((t) => t.id === AUDIO_TRACK_ID)?.clips ?? [],

      addClip: async (partial) => {
        const trackId = partial.trackId ?? MAIN_TRACK_ID
        const id = partial.id ?? newId()
        let clip: TimelineClip = {
          id,
          mediaType: partial.mediaType,
          src: partial.src,
          duration: partial.duration ?? (partial.mediaType === 'image' ? 3 : 5),
          trimStart: partial.trimStart ?? 0,
          trimEnd: partial.trimEnd ?? 0,
          thumbnailDataUrl: partial.thumbnailDataUrl,
          sourceNodeId: partial.sourceNodeId,
          label: partial.label,
        }
        clip = await fillThumbnail(clip)
        set((s) => ({
          tracks: patchTracks(s.tracks, trackId, (clips) => [...clips, clip]),
        }))
        return id
      },

      insertClipAfter: async (trackId, afterId, partial) => {
        const id = partial.id ?? newId()
        let clip: TimelineClip = {
          id,
          mediaType: partial.mediaType,
          src: partial.src,
          duration: partial.duration ?? (partial.mediaType === 'image' ? 3 : 5),
          trimStart: partial.trimStart ?? 0,
          trimEnd: partial.trimEnd ?? 0,
          thumbnailDataUrl: partial.thumbnailDataUrl,
          sourceNodeId: partial.sourceNodeId,
          label: partial.label,
        }
        clip = await fillThumbnail(clip)
        set((s) => ({
          tracks: patchTracks(s.tracks, trackId, (clips) => {
            const idx = clips.findIndex((c) => c.id === afterId)
            if (idx < 0) return [...clips, clip]
            const next = [...clips]
            next.splice(idx + 1, 0, clip)
            return next
          }),
        }))
        return id
      },

      removeClip: (trackId, clipId) =>
        set((s) => ({
          tracks: patchTracks(s.tracks, trackId, (clips) =>
            clips.filter((c) => c.id !== clipId),
          ),
          selectedClipId: s.selectedClipId === clipId ? null : s.selectedClipId,
        })),

      reorderClips: (trackId, orderedIds) =>
        set((s) => ({
          tracks: patchTracks(s.tracks, trackId, (clips) => {
            const map = new Map(clips.map((c) => [c.id, c]))
            const next = orderedIds.map((id) => map.get(id)).filter(Boolean) as TimelineClip[]
            const rest = clips.filter((c) => !orderedIds.includes(c.id))
            return [...next, ...rest]
          }),
        })),

      updateClip: (trackId, clipId, patch) =>
        set((s) => ({
          tracks: patchTracks(s.tracks, trackId, (clips) =>
            clips.map((c) => (c.id === clipId ? { ...c, ...patch } : c)),
          ),
        })),

      setPlayheadSec: (playheadSec) => set({ playheadSec }),
      setPxPerSec: (pxPerSec) => set({ pxPerSec: Math.max(8, Math.min(200, pxPerSec)) }),
      setPlaying: (isPlaying) => set({ isPlaying }),

      clearClips: () =>
        set({
          tracks: createDefaultTracks(),
          playheadSec: 0,
          isPlaying: false,
          selectedClipId: null,
        }),
    }),
    {
      name: 'infinite-canvas-timeline',
      version: 2,
      partialize: (s) => ({
        tracks: stripBlobClips(s.tracks),
        pxPerSec: s.pxPerSec,
      }),
      migrate: (persisted, fromVersion) => {
        const p = persisted as Record<string, unknown> & {
          clips?: TimelineClip[]
          tracks?: TimelineTrack[]
          pxPerSec?: number
        }
        if (fromVersion < 2 && Array.isArray(p.clips)) {
          return {
            tracks: [
              {
                id: MAIN_TRACK_ID,
                kind: 'main' as const,
                label: 'Video',
                clips: p.clips,
              },
              {
                id: AUDIO_TRACK_ID,
                kind: 'audio' as const,
                label: 'Audio',
                clips: [],
              },
            ],
            pxPerSec: typeof p.pxPerSec === 'number' ? p.pxPerSec : 40,
          }
        }
        if (!p.tracks || !Array.isArray(p.tracks)) {
          return {
            tracks: createDefaultTracks(),
            pxPerSec: typeof p.pxPerSec === 'number' ? p.pxPerSec : 40,
          }
        }
        return { tracks: p.tracks, pxPerSec: typeof p.pxPerSec === 'number' ? p.pxPerSec : 40 }
      },
    },
  ),
)

/** Main-track clip + local playback time in source (for preview / seek). */
export function getMainTrackClipAtTime(
  tracks: TimelineTrack[],
  t: number,
): { clip: TimelineClip; localT: number } | null {
  const main = tracks.find((tr) => tr.id === MAIN_TRACK_ID)
  if (!main) return null
  let acc = 0
  for (const c of main.clips) {
    const len = effectiveClipDuration(c)
    if (t < acc + len) {
      const offset = t - acc
      if (c.mediaType === 'video' || c.mediaType === 'audio') {
        return { clip: c, localT: c.trimStart + offset }
      }
      return { clip: c, localT: offset }
    }
    acc += len
  }
  return null
}
