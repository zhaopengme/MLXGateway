export type NodeKind =
  | 'media'
  | 'text'
  | 'gen-image'
  | 'gen-image-advanced'
  | 'gen-video'
  | 'chat'
  | 'tts'
  | 'stt'
  | 'prompt'

export type GenStatus = 'idle' | 'generating' | 'done' | 'error'

export type ChatMsg = { role: 'system' | 'user' | 'assistant'; content: string }

export interface CanvasNodeData {
  title?: string
  /** Media URL (http, blob, data) or text body */
  content?: string
  prompt?: string
  previewType?: 'image' | 'video'

  /** MLX generation lifecycle */
  status?: GenStatus
  error?: string

  /** gen-image / gen-video / chat / tts / stt */
  model?: string
  imageSize?: string

  /** Video params (also in settings for persistence) */
  videoWidth?: number
  videoHeight?: number
  numFrames?: number
  fps?: number
  pipeline?: string
  cfgScale?: number
  audioCfgScale?: number
  imageStrength?: number
  tiling?: string
  textEncoderRepo?: string | null

  /** Chat node */
  chatMessages?: ChatMsg[]
  chatInputDraft?: string
  chatPartial?: string
  temperature?: number

  /** TTS */
  voice?: string
  ttsFormat?: string
  /** Object URL for last TTS output (browser session) */
  audioBlobUrl?: string

  /** STT */
  sttModel?: string
  transcribedText?: string

  /** gen-image: optional edit from file / connection */
  useImageEdit?: boolean
}

export interface CanvasNode {
  id: string
  type: NodeKind
  x: number
  y: number
  width: number
  height: number
  data: CanvasNodeData
  settings: Record<string, unknown>
}

export interface Connection {
  id: string
  from: string
  to: string
  fromPort?: string
  toPort?: string
}

export interface ViewState {
  x: number
  y: number
  zoom: number
}

export type ToolMode = 'select' | 'pan'
