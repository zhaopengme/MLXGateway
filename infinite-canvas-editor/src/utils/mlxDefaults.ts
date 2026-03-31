/** Defaults aligned with [MLXGateway README](../../README.md) */
export const DEFAULT_IMAGE_MODEL = 'black-forest-labs/FLUX.2-klein-4B'
export const DEFAULT_IMAGE_EDIT_MODEL = 'flux2-klein-4b-edit'
export const DEFAULT_VIDEO_MODEL = 'prince-canuma/LTX-2.3-distilled'
export const DEFAULT_CHAT_MODEL = 'mlx-community/Qwen3-4B-Instruct-2507-4bit'
export const DEFAULT_TTS_MODEL = 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16'
/** User should pick from GET /v1/models when available */
export const DEFAULT_STT_MODEL = 'mlx-community/whisper-large-v3-turbo-asr-fp16'

export const VIDEO_PIPELINES = [
  'distilled',
  'dev',
  'dev-two-stage',
  'dev-two-stage-hq',
] as const

/** Duration options in seconds for the video generation UI */
export const VIDEO_DURATION_OPTIONS = [2, 3, 4, 5, 10, 15, 20] as const

/**
 * Convert a duration in seconds to a valid LTX-Video num_frames value.
 * LTX requires num_frames = 8n + 1 (i.e. 9, 17, 25, … 257).
 * Formula: raw = sec * fps + 1, then round to nearest 8n + 1.
 */
export function durationToFrames(sec: number, fps: number): number {
  const raw = sec * fps + 1
  const n = Math.round((raw - 1) / 8)
  return Math.max(9, n * 8 + 1)
}

/**
 * Convert a num_frames value back to duration in seconds.
 * Formula: duration = (num_frames - 1) / fps
 */
export function framesToDuration(frames: number, fps: number): number {
  return (frames - 1) / fps
}

export const IMAGE_SIZES = ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792'] as const
