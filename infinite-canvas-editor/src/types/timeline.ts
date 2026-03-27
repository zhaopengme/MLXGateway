export type ClipMediaType = 'video' | 'image' | 'audio'

export type TrackKind = 'main' | 'audio'

export const MAIN_TRACK_ID = 'track-main'
export const AUDIO_TRACK_ID = 'track-audio'

export interface TimelineClip {
  id: string
  mediaType: ClipMediaType
  /** Blob URL or remote URL */
  src: string
  /** Source duration in seconds (video/audio); for image, hold duration on timeline */
  duration: number
  /** Trim from start of source (seconds) — video / audio */
  trimStart: number
  /** Trim from end of source — video / audio */
  trimEnd: number
  thumbnailDataUrl?: string
  /** Reference back to canvas node if spawned from canvas */
  sourceNodeId?: string
  label?: string
}

export interface TimelineTrack {
  id: string
  kind: TrackKind
  label: string
  clips: TimelineClip[]
}
