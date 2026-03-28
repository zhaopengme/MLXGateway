/** Defaults aligned with [MLXGateway README](../../README.md) */
export const DEFAULT_IMAGE_MODEL = 'black-forest-labs/FLUX.2-klein-4B'
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

export const VIDEO_NUM_FRAMES_OPTIONS = [
  9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177,
  185, 193, 201, 209, 217, 225, 233, 241, 249, 257,
]

export const IMAGE_SIZES = ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792'] as const
