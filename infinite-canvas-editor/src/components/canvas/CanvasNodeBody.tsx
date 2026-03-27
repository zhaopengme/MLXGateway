import { Loader2, MessageSquare, Mic, Send, SendToBack, Sparkles, Volume2 } from 'lucide-react'
import { useRef, useState } from 'react'
import * as api from '../../api/gateway'
import { useCanvasStore, DEFAULT_IMAGE_MODEL, DEFAULT_VIDEO_MODEL } from '../../stores/canvasStore'
import { useImageEditStore } from '../../stores/imageEditStore'
import { useProjectStore } from '../../stores/projectStore'
import { useTimelineStore } from '../../stores/timelineStore'
import type { CanvasNode as NodeType } from '../../types/canvas'
import {
  blobUrlToBase64,
  getImageUrlFromNode,
  getOrderedIncomingNodes,
  getPromptFromIncoming,
  isImageLikeNode,
} from '../../utils/connectionInputs'
import { DEFAULT_CHAT_MODEL, DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL } from '../../utils/mlxDefaults'
import { getAudioDurationSec, getVideoDurationSec } from '../../utils/mediaDuration'
import { AUDIO_TRACK_ID } from '../../types/timeline'

type Props = {
  node: NodeType
  isDark: boolean
}

async function urlToImageFile(url: string, filename: string): Promise<File> {
  const res = await fetch(url)
  const blob = await res.blob()
  return new File([blob], filename, { type: blob.type || 'image/png' })
}

export function CanvasNodeBody({ node, isDark }: Props) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const openImageEditor = useImageEditStore((s) => s.openEditor)

  if (node.type === 'media' || node.type === 'gen-video' || node.type === 'gen-image') {
    if (node.type === 'gen-image') {
      return <GenImageBody node={node} isDark={isDark} />
    }
    if (node.type === 'gen-video') {
      return <GenVideoBody node={node} isDark={isDark} />
    }
    const url = node.data.content
    const isVid = node.data.previewType === 'video'
    if (!url) {
      return (
        <div className={`p-4 text-xs ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
          No media URL — paste a link in properties or connect a generator.
        </div>
      )
    }
    return (
      <div
        className="w-full h-full flex items-center justify-center bg-black/40"
        draggable
        title="Drag to timeline · Double-click image to crop / compress"
        onDragStart={(e) => {
          e.stopPropagation()
          e.dataTransfer.setData('application/x-canvas-node', node.id)
          e.dataTransfer.effectAllowed = 'copy'
        }}
      >
        {isVid ? (
          <video
            src={url}
            className="max-w-full max-h-full object-contain pointer-events-auto"
            controls
            muted
            playsInline
          />
        ) : (
          <img
            src={url}
            alt=""
            className="max-w-full max-h-full object-contain cursor-pointer"
            draggable={false}
            onDoubleClick={(e) => {
              e.stopPropagation()
              openImageEditor(node.id, url)
            }}
          />
        )}
      </div>
    )
  }

  if (node.type === 'chat') {
    return <ChatNodeBody node={node} isDark={isDark} />
  }
  if (node.type === 'tts') {
    return <TtsNodeBody node={node} isDark={isDark} />
  }
  if (node.type === 'stt') {
    return <SttNodeBody node={node} isDark={isDark} />
  }

  if (node.type === 'text' || node.type === 'prompt') {
    return (
      <textarea
        value={node.data.content ?? ''}
        onChange={(e) =>
          updateNode(node.id, { data: { ...node.data, content: e.target.value } })
        }
        className={`w-full h-full resize-none border-0 p-3 text-sm outline-none ${
          isDark ? 'bg-zinc-900 text-zinc-100 placeholder:text-zinc-600' : 'bg-white text-zinc-900'
        }`}
        placeholder="Notes / prompt…"
      />
    )
  }

  return null
}

const btnPrimary = `px-2 py-1 rounded text-[10px] font-medium flex items-center gap-1 justify-center`

function GenImageBody({ node, isDark }: { node: NodeType; isDark: boolean }) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const openImageEditor = useImageEditStore((s) => s.openEditor)
  const addHistoryItem = useProjectStore((s) => s.addHistoryItem)
  const addClip = useTimelineStore((s) => s.addClip)
  const incoming = getOrderedIncomingNodes(node.id)
  const imageRefs = incoming.filter(isImageLikeNode)
  const canEdit = node.data.useImageEdit && imageRefs.length > 0

  const onGenerate = async () => {
    const prompt = getPromptFromIncoming(incoming, node.data.prompt)
    if (!prompt.trim()) {
      updateNode(node.id, { data: { ...node.data, status: 'error', error: 'Prompt is empty' } })
      return
    }
    updateNode(node.id, { data: { ...node.data, status: 'generating', error: undefined } })
    try {
      let urls: { url: string }[]
      if (canEdit) {
        const files: File[] = []
        let i = 0
        for (const src of imageRefs) {
          const u = getImageUrlFromNode(src)
          if (!u) continue
          files.push(await urlToImageFile(u, `ref_${i++}.png`))
        }
        if (!files.length) throw new Error('No usable image URLs for edit')
        urls = await api.editImage({
          model: node.data.model ?? 'flux2-klein-9b-edit',
          prompt,
          images: files,
          response_format: 'url',
        })
      } else {
        urls = await api.generateImage({
          model: node.data.model ?? DEFAULT_IMAGE_MODEL,
          prompt,
          size: node.data.imageSize ?? '1024x1024',
          n: 1,
        })
      }
      const first = urls[0]?.url
      if (!first) throw new Error('No image URL in response')
      updateNode(node.id, {
        data: {
          ...node.data,
          status: 'done',
          content: first,
          previewType: 'image',
          error: undefined,
        },
      })
      addHistoryItem({ type: 'image', url: first, prompt, label: node.data.title })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Generation failed'
      updateNode(node.id, { data: { ...node.data, status: 'error', error: msg } })
    }
  }

  const url = node.data.content
  const loading = node.data.status === 'generating'
  const err = node.data.error

  return (
    <div className={`flex flex-col h-full min-h-0 ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      <div className="shrink-0 flex flex-wrap gap-1 p-2 border-b border-zinc-700/50">
        <button
          type="button"
          disabled={loading}
          onClick={() => void onGenerate()}
          className={`${btnPrimary} ${isDark ? 'bg-violet-600 text-white' : 'bg-violet-600 text-white'}`}
        >
          {loading ? <Loader2 className="size-3 animate-spin" /> : <Sparkles className="size-3" />}
          {canEdit ? 'Edit' : 'Generate'}
        </button>
        <button
          type="button"
          disabled={!url || loading}
          onClick={() =>
            void addClip({
              mediaType: 'image',
              src: url!,
              duration: 3,
              trimStart: 0,
              trimEnd: 0,
              label: node.data.title || 'image',
              sourceNodeId: node.id,
            })
          }
          className={`${btnPrimary} ${isDark ? 'bg-zinc-700 text-zinc-200' : 'bg-zinc-200 text-zinc-800'}`}
        >
          <SendToBack className="size-3" />
          Timeline
        </button>
      </div>
      {incoming.length > 0 && (
        <div className={`text-[9px] px-2 py-1 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
          {imageRefs.length} image input(s) from graph
          {node.data.useImageEdit ? ' — using /v1/images/edits' : ''}
        </div>
      )}
      {err && <div className="text-[10px] text-red-400 px-2 py-1 break-words">{err}</div>}
      {url ? (
        <div
          className="flex-1 min-h-0 flex items-center justify-center bg-black/40"
          draggable
          title="Drag to timeline · Double-click image to crop / compress"
          onDragStart={(e) => {
            e.stopPropagation()
            e.dataTransfer.setData('application/x-canvas-node', node.id)
            e.dataTransfer.effectAllowed = 'copy'
          }}
        >
          <img
            src={url}
            alt=""
            className="max-w-full max-h-full object-contain cursor-pointer"
            draggable={false}
            onDoubleClick={(e) => {
              e.stopPropagation()
              openImageEditor(node.id, url)
            }}
          />
        </div>
      ) : (
        <div className={`flex-1 flex items-center justify-center text-xs p-4 ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
          {loading ? 'Generating…' : 'Run Generate or set URL in properties'}
        </div>
      )}
    </div>
  )
}

function GenVideoBody({ node, isDark }: { node: NodeType; isDark: boolean }) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const addHistoryItem = useProjectStore((s) => s.addHistoryItem)
  const addClip = useTimelineStore((s) => s.addClip)
  const incoming = getOrderedIncomingNodes(node.id)

  const onGenerate = async () => {
    const prompt = getPromptFromIncoming(incoming, node.data.prompt)
    if (!prompt.trim()) {
      updateNode(node.id, { data: { ...node.data, status: 'error', error: 'Prompt is empty' } })
      return
    }
    updateNode(node.id, { data: { ...node.data, status: 'generating', error: undefined } })
    try {
      const imgs = incoming.filter(isImageLikeNode).map(getImageUrlFromNode).filter(Boolean) as string[]
      const ttsNode = incoming.find((n) => n.type === 'tts' && n.data.audioBlobUrl)
      let audio_file: string | undefined
      if (ttsNode?.data.audioBlobUrl) {
        audio_file = await blobUrlToBase64(ttsNode.data.audioBlobUrl)
      }
      const body: Record<string, unknown> = {
        prompt,
        model: node.data.model ?? DEFAULT_VIDEO_MODEL,
        width: node.data.videoWidth ?? 512,
        height: node.data.videoHeight ?? 512,
        num_frames: node.data.numFrames ?? 97,
        fps: node.data.fps ?? 24,
        pipeline: node.data.pipeline ?? 'distilled',
        response_format: 'url',
        cfg_scale: node.data.cfgScale ?? 3,
        audio_cfg_scale: node.data.audioCfgScale ?? 7,
        image_strength: node.data.imageStrength ?? 1,
        tiling: node.data.tiling ?? 'auto',
      }
      if (node.data.textEncoderRepo != null && node.data.textEncoderRepo !== '')
        body.text_encoder_repo = node.data.textEncoderRepo
      if (imgs[0]) body.image_url = imgs[0]
      if (imgs[1]) body.end_image_url = imgs[1]
      if (audio_file) body.audio_file = audio_file

      const { url } = await api.generateVideo(body)
      updateNode(node.id, {
        data: {
          ...node.data,
          status: 'done',
          content: url,
          previewType: 'video',
          error: undefined,
        },
      })
      addHistoryItem({ type: 'video', url, prompt, label: node.data.title })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Video generation failed'
      updateNode(node.id, { data: { ...node.data, status: 'error', error: msg } })
    }
  }

  const url = node.data.content
  const loading = node.data.status === 'generating'
  const err = node.data.error

  return (
    <div className={`flex flex-col h-full min-h-0 ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      <div className="shrink-0 flex flex-wrap gap-1 p-2 border-b border-zinc-700/50">
        <button
          type="button"
          disabled={loading}
          onClick={() => void onGenerate()}
          className={`${btnPrimary} ${isDark ? 'bg-sky-600 text-white' : 'bg-sky-600 text-white'}`}
        >
          {loading ? <Loader2 className="size-3 animate-spin" /> : <Sparkles className="size-3" />}
          Generate video
        </button>
        <button
          type="button"
          disabled={loading || !url}
          onClick={() => {
            if (!url) return
            void (async () => {
              const duration = await getVideoDurationSec(url)
              await addClip({
                mediaType: 'video',
                src: url,
                duration,
                trimStart: 0,
                trimEnd: 0,
                label: node.data.title || 'video',
                sourceNodeId: node.id,
              })
            })()
          }}
          className={`${btnPrimary} ${isDark ? 'bg-zinc-700 text-zinc-200' : 'bg-zinc-200 text-zinc-800'}`}
        >
          <SendToBack className="size-3" />
          Timeline
        </button>
      </div>
      {incoming.length > 0 && (
        <div className={`text-[9px] px-2 py-1 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
          Inputs:{' '}
          {incoming
            .filter(isImageLikeNode)
            .length.toString()} image(s),{' '}
          {incoming.some((n) => n.type === 'tts') ? 'TTS' : 'no TTS'}
        </div>
      )}
      {err && <div className="text-[10px] text-red-400 px-2 py-1 break-words">{err}</div>}
      {url ? (
        <div
          className="flex-1 min-h-0 flex items-center justify-center bg-black/40"
          draggable
          title="Drag to timeline"
          onDragStart={(e) => {
            e.stopPropagation()
            e.dataTransfer.setData('application/x-canvas-node', node.id)
            e.dataTransfer.effectAllowed = 'copy'
          }}
        >
          <video
            src={url}
            className="max-w-full max-h-full object-contain pointer-events-auto"
            controls
            muted
            playsInline
          />
        </div>
      ) : (
        <div className={`flex-1 flex items-center justify-center text-xs p-4 ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
          {loading ? 'Generating (GPU may take minutes)…' : 'Configure prompt & params, then Generate'}
        </div>
      )}
    </div>
  )
}

function ChatNodeBody({ node, isDark }: { node: NodeType; isDark: boolean }) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [streaming, setStreaming] = useState(false)
  const messages = node.data.chatMessages ?? []
  const draft = node.data.chatInputDraft ?? ''

  const scrollBottom = () => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
    })
  }

  const send = async () => {
    const text = draft.trim()
    if (!text || streaming) return
    const model = node.data.model ?? DEFAULT_CHAT_MODEL
    const userMsg = { role: 'user' as const, content: text }
    const baseMsgs = [...messages, userMsg]
    updateNode(node.id, {
      data: {
        ...node.data,
        chatMessages: baseMsgs,
        chatInputDraft: '',
        chatPartial: '',
      },
    })
    scrollBottom()
    setStreaming(true)
    let assistant = ''
    try {
      await api.chatCompletionsStream(
        {
          model,
          messages: baseMsgs,
          temperature: node.data.temperature ?? 0.7,
        },
        (chunk) => {
          assistant += chunk
          const cur = useCanvasStore.getState().nodes.find((x) => x.id === node.id)
          const d = cur?.data ?? node.data
          updateNode(node.id, {
            data: {
              ...d,
              chatMessages: [...baseMsgs, { role: 'assistant', content: assistant }],
              chatPartial: assistant,
            },
          })
          scrollBottom()
        },
      )
      const cur = useCanvasStore.getState().nodes.find((x) => x.id === node.id)
      const d = cur?.data ?? node.data
      updateNode(node.id, {
        data: {
          ...d,
          chatMessages: [...baseMsgs, { role: 'assistant', content: assistant }],
          chatPartial: undefined,
          error: undefined,
        },
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Chat failed'
      const cur = useCanvasStore.getState().nodes.find((x) => x.id === node.id)
      const d = cur?.data ?? node.data
      updateNode(node.id, {
        data: {
          ...d,
          chatMessages: baseMsgs,
          error: msg,
          chatPartial: undefined,
        },
      })
    } finally {
      setStreaming(false)
      scrollBottom()
    }
  }

  return (
    <div className={`flex flex-col h-full min-h-0 text-[11px] ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      <div className="shrink-0 flex items-center gap-1 px-2 py-1 border-b border-zinc-700/50">
        <MessageSquare className="size-3 text-zinc-500" />
        <span className="opacity-70">Chat</span>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2"
      >
        {messages.length === 0 && (
          <div className={isDark ? 'text-zinc-500' : 'text-zinc-400'}>Send a message…</div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-lg px-2 py-1 whitespace-pre-wrap ${
              m.role === 'user'
                ? isDark
                  ? 'bg-blue-900/40 ml-4'
                  : 'bg-blue-50 ml-4'
                : isDark
                  ? 'bg-zinc-800 mr-4'
                  : 'bg-zinc-100 mr-4'
            }`}
          >
            {m.content || (streaming && i === messages.length - 1 ? '…' : '')}
          </div>
        ))}
      </div>
      {node.data.error && (
        <div className="text-[10px] text-red-400 px-2">{node.data.error}</div>
      )}
      <div className="shrink-0 flex gap-1 p-2 border-t border-zinc-700/50">
        <input
          value={draft}
          onChange={(e) =>
            updateNode(node.id, { data: { ...node.data, chatInputDraft: e.target.value } })
          }
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), void send())}
          placeholder="Message…"
          disabled={streaming}
          className={`flex-1 min-w-0 px-2 py-1 rounded border text-[11px] ${
            isDark ? 'bg-zinc-800 border-zinc-600 text-zinc-100' : 'bg-white border-zinc-300'
          }`}
        />
        <button
          type="button"
          disabled={streaming || !draft.trim()}
          onClick={() => void send()}
          className={`${btnPrimary} shrink-0 ${isDark ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'}`}
        >
          {streaming ? <Loader2 className="size-3 animate-spin" /> : <Send className="size-3" />}
        </button>
      </div>
    </div>
  )
}

function TtsNodeBody({ node, isDark }: { node: NodeType; isDark: boolean }) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const addClip = useTimelineStore((s) => s.addClip)
  const incoming = getOrderedIncomingNodes(node.id)
  const text =
    node.data.prompt?.trim() ||
    getPromptFromIncoming(incoming, undefined) ||
    node.data.content ||
    ''
  const [busy, setBusy] = useState(false)

  const speak = async () => {
    const input = text.trim()
    if (!input) return
    if (node.data.audioBlobUrl) {
      URL.revokeObjectURL(node.data.audioBlobUrl)
    }
    setBusy(true)
    try {
      const blob = await api.textToSpeech({
        model: node.data.model ?? DEFAULT_TTS_MODEL,
        input,
        voice: node.data.voice ?? 'af_sky',
        response_format: node.data.ttsFormat ?? 'wav',
      })
      const url = URL.createObjectURL(blob)
      updateNode(node.id, {
        data: { ...node.data, audioBlobUrl: url, status: 'done', error: undefined },
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'TTS failed'
      updateNode(node.id, { data: { ...node.data, status: 'error', error: msg } })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={`flex flex-col h-full min-h-0 p-2 gap-2 ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      <div className="flex items-center gap-1">
        <Volume2 className="size-3 text-zinc-500" />
        <span className="text-[10px] opacity-70">Text-to-speech</span>
      </div>
      <textarea
        value={node.data.prompt ?? ''}
        onChange={(e) => updateNode(node.id, { data: { ...node.data, prompt: e.target.value } })}
        placeholder="Text to speak…"
        rows={3}
        className={`w-full resize-none rounded border text-[11px] p-2 ${
          isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'
        }`}
      />
      <div className="flex flex-wrap gap-1">
        <button
          type="button"
          disabled={busy || !text.trim()}
          onClick={() => void speak()}
          className={`${btnPrimary} ${isDark ? 'bg-amber-600 text-white' : 'bg-amber-600 text-white'}`}
        >
          {busy ? <Loader2 className="size-3 animate-spin" /> : null}
          Speak
        </button>
        <button
          type="button"
          disabled={!node.data.audioBlobUrl || busy}
          onClick={() => {
            const u = node.data.audioBlobUrl
            if (!u) return
            void (async () => {
              const duration = await getAudioDurationSec(u)
              await addClip({
                mediaType: 'audio',
                src: u,
                duration,
                trimStart: 0,
                trimEnd: 0,
                label: node.data.title || 'tts',
                sourceNodeId: node.id,
                trackId: AUDIO_TRACK_ID,
              })
            })()
          }}
          className={`${btnPrimary} ${isDark ? 'bg-zinc-700 text-zinc-200' : 'bg-zinc-200 text-zinc-800'}`}
        >
          <SendToBack className="size-3" />
          Timeline
        </button>
      </div>
      {node.data.error && <div className="text-[10px] text-red-400">{node.data.error}</div>}
      {node.data.audioBlobUrl && (
        <audio src={node.data.audioBlobUrl} controls className="w-full" />
      )}
    </div>
  )
}

function SttNodeBody({ node, isDark }: { node: NodeType; isDark: boolean }) {
  const updateNode = useCanvasStore((s) => s.updateNode)
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)

  const run = async (file: File) => {
    setBusy(true)
    updateNode(node.id, { data: { ...node.data, status: 'generating', error: undefined } })
    try {
      const t = await api.speechToText(file, node.data.sttModel ?? DEFAULT_STT_MODEL)
      updateNode(node.id, {
        data: { ...node.data, transcribedText: t, status: 'done', content: t },
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'STT failed'
      updateNode(node.id, { data: { ...node.data, status: 'error', error: msg } })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={`flex flex-col h-full min-h-0 p-2 gap-2 ${isDark ? 'bg-zinc-900' : 'bg-white'}`}>
      <div className="flex items-center gap-1">
        <Mic className="size-3 text-zinc-500" />
        <span className="text-[10px] opacity-70">Speech-to-text</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (f) void run(f)
        }}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className={`${btnPrimary} ${isDark ? 'bg-emerald-700 text-white' : 'bg-emerald-600 text-white'}`}
      >
        {busy ? <Loader2 className="size-3 animate-spin" /> : <Mic className="size-3" />}
        Upload audio
      </button>
      {node.data.error && <div className="text-[10px] text-red-400">{node.data.error}</div>}
      <div
        className={`flex-1 min-h-0 overflow-y-auto rounded border p-2 text-[11px] whitespace-pre-wrap ${
          isDark ? 'border-zinc-700 bg-zinc-950' : 'border-zinc-200 bg-zinc-50'
        }`}
      >
        {node.data.transcribedText || (busy ? 'Transcribing…' : '—')}
      </div>
    </div>
  )
}
