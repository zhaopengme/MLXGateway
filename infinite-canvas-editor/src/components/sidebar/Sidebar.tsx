import { History, ImagePlus, Layers, MessageSquare, Mic, Sparkles, Type, Video, Volume2, Wand2 } from 'lucide-react'
import { useState } from 'react'
import { useCanvasStore } from '../../stores/canvasStore'
import { useProjectStore } from '../../stores/projectStore'
import {
  DEFAULT_CHAT_MODEL,
  DEFAULT_IMAGE_MODEL,
  DEFAULT_STT_MODEL,
  DEFAULT_TTS_MODEL,
  DEFAULT_VIDEO_MODEL,
} from '../../utils/mlxDefaults'

type Props = { isDark: boolean }

export function Sidebar({ isDark }: Props) {
  const addNode = useCanvasStore((s) => s.addNode)
  const history = useProjectStore((s) => s.history)
  const [historyOpen, setHistoryOpen] = useState(true)

  const rail = `w-12 shrink-0 flex flex-col items-center py-2 gap-1 border-r ${
    isDark ? 'bg-zinc-900 border-zinc-800' : 'bg-white border-zinc-200'
  }`

  const btn = `p-2 rounded-lg transition-colors ${
    isDark ? 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200' : 'text-zinc-600 hover:bg-zinc-100'
  }`

  return (
    <div className="flex shrink-0 h-full min-h-0">
      <div className={rail}>
        <button type="button" className={btn} title="Add media node" onClick={() => addSquareNode('media')}>
          <ImagePlus className="size-5" />
        </button>
        <button type="button" className={btn} title="Add video gen node" onClick={() => addSquareNode('gen-video')}>
          <Video className="size-5" />
        </button>
        <button type="button" className={btn} title="Add image gen node" onClick={() => addSquareNode('gen-image')}>
          <Wand2 className="size-5" />
        </button>
        <button type="button" className={btn} title="Add advanced image gen (with references)" onClick={() => addSquareNode('gen-image-advanced')}>
          <Layers className="size-5" />
        </button>
        <button type="button" className={btn} title="Add chat node" onClick={() => addSquareNode('chat')}>
          <MessageSquare className="size-5" />
        </button>
        <button type="button" className={btn} title="Add TTS node" onClick={() => addSquareNode('tts')}>
          <Volume2 className="size-5" />
        </button>
        <button type="button" className={btn} title="Add STT node" onClick={() => addSquareNode('stt')}>
          <Mic className="size-5" />
        </button>
        <button type="button" className={btn} title="Add text note" onClick={() => addSquareNode('text')}>
          <Type className="size-5" />
        </button>
        <button type="button" className={btn} title="Add prompt" onClick={() => addSquareNode('prompt')}>
          <Sparkles className="size-5" />
        </button>
        <div className="flex-1 min-h-2" />
        <button
          type="button"
          className={`${btn} ${historyOpen ? 'bg-blue-600/20 text-blue-400' : ''}`}
          title="History"
          onClick={() => setHistoryOpen(!historyOpen)}
        >
          <History className="size-5" />
        </button>
      </div>
      {historyOpen && (
        <div
          className={`w-56 flex flex-col min-h-0 border-r ${
            isDark ? 'bg-zinc-900/95 border-zinc-800' : 'bg-zinc-50 border-zinc-200'
          }`}
        >
          <div
            className={`px-3 py-2 text-[10px] font-semibold uppercase tracking-wider border-b ${
              isDark ? 'border-zinc-800 text-zinc-500' : 'border-zinc-200 text-zinc-500'
            }`}
          >
            Library
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {history.map((item) => (
              <div
                key={item.id}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData(
                    'application/x-history-item',
                    JSON.stringify({ type: item.type, url: item.url, prompt: item.prompt }),
                  )
                  e.dataTransfer.effectAllowed = 'copy'
                }}
                className={`rounded-lg border p-2 cursor-grab active:cursor-grabbing ${
                  isDark ? 'border-zinc-700 bg-zinc-800/50' : 'border-zinc-200 bg-white'
                }`}
              >
                <div className={`text-[10px] font-medium truncate ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
                  {item.prompt?.slice(0, 40) || item.type}
                </div>
                <div className={`text-[9px] mt-0.5 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
                  Drag to timeline
                </div>
                {item.type === 'video' ? (
                  <video src={item.url} className="mt-1 w-full h-14 object-cover rounded" muted preload="metadata" />
                ) : (
                  <img src={item.url} alt="" className="mt-1 w-full h-14 object-cover rounded" />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  function addSquareNode(
    kind:
      | 'media'
      | 'gen-video'
      | 'gen-image'
      | 'gen-image-advanced'
      | 'chat'
      | 'tts'
      | 'stt'
      | 'text'
      | 'prompt',
  ) {
    const base = { x: 120 + Math.random() * 40, y: 100 + Math.random() * 40, width: 280, height: 200, settings: {} }
    if (kind === 'text' || kind === 'prompt') {
      addNode({
        ...base,
        height: 140,
        type: kind,
        data: { title: kind === 'prompt' ? 'Prompt' : 'Note', content: '' },
      })
      return
    }
    if (kind === 'media') {
      addNode({
        ...base,
        type: 'media',
        data: { title: 'Media', content: '', previewType: 'image' },
      })
      return
    }
    if (kind === 'gen-image') {
      addNode({
        ...base,
        height: 240,
        type: 'gen-image',
        data: {
          title: 'Image gen',
          prompt: '',
          previewType: 'image',
          content: '',
          model: DEFAULT_IMAGE_MODEL,
          imageSize: '1024x1024',
          status: 'idle',
        },
        settings: {},
      })
      return
    }
    if (kind === 'gen-image-advanced') {
      addNode({
        ...base,
        height: 280,
        type: 'gen-image-advanced',
        data: {
          title: 'Image gen+',
          prompt: '',
          previewType: 'image',
          content: '',
          model: DEFAULT_IMAGE_MODEL,
          imageSize: '1024x1024',
          status: 'idle',
        },
        settings: {},
      })
      return
    }
    if (kind === 'chat') {
      addNode({
        ...base,
        width: 320,
        height: 360,
        type: 'chat',
        data: {
          title: 'Chat',
          model: DEFAULT_CHAT_MODEL,
          chatMessages: [],
          chatInputDraft: '',
          temperature: 0.7,
        },
        settings: {},
      })
      return
    }
    if (kind === 'tts') {
      addNode({
        ...base,
        height: 260,
        type: 'tts',
        data: {
          title: 'TTS',
          model: DEFAULT_TTS_MODEL,
          voice: 'liuyifei',
          ttsFormat: 'wav',
          prompt: '',
          status: 'idle',
        },
        settings: {},
      })
      return
    }
    if (kind === 'stt') {
      addNode({
        ...base,
        height: 280,
        type: 'stt',
        data: {
          title: 'STT',
          sttModel: DEFAULT_STT_MODEL,
          status: 'idle',
        },
        settings: {},
      })
      return
    }
    addNode({
      ...base,
      height: 260,
      type: 'gen-video',
      data: {
        title: 'Video gen',
        prompt: '',
        previewType: 'video',
        content: '',
        model: DEFAULT_VIDEO_MODEL,
        status: 'idle',
        videoWidth: 512,
        videoHeight: 512,
        numFrames: 97,
        fps: 24,
        pipeline: 'distilled',
        cfgScale: 3,
        audioCfgScale: 7,
        imageStrength: 1,
        tiling: 'auto',
      },
      settings: {},
    })
  }
}
