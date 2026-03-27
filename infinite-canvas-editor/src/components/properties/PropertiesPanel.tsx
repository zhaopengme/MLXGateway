import { Link2, Trash2 } from 'lucide-react'
import { useGatewayStore } from '../../stores/gatewayStore'
import { useCanvasStore } from '../../stores/canvasStore'
import { useTimelineStore } from '../../stores/timelineStore'
import {
  DEFAULT_CHAT_MODEL,
  DEFAULT_IMAGE_MODEL,
  DEFAULT_STT_MODEL,
  DEFAULT_TTS_MODEL,
  DEFAULT_VIDEO_MODEL,
  IMAGE_SIZES,
  VIDEO_NUM_FRAMES_OPTIONS,
  VIDEO_PIPELINES,
} from '../../utils/mlxDefaults'

type Props = { isDark: boolean }

const MIN_VISIBLE = 0.15

function parseTimelinePositive(raw: string): number | null {
  const v = parseFloat(raw)
  return Number.isFinite(v) ? v : null
}

function clampTrimsToDuration(duration: number, trimStart: number, trimEnd: number) {
  const maxPair = Math.max(0, duration - MIN_VISIBLE)
  let ts = Math.max(0, trimStart)
  let te = Math.max(0, trimEnd)
  if (ts + te > maxPair) {
    te = Math.max(0, maxPair - ts)
    if (ts + te > maxPair) ts = Math.max(0, maxPair - te)
  }
  return { trimStart: ts, trimEnd: te }
}

function ModelPicker({
  value,
  onChange,
  isDark,
  filter,
}: {
  value: string
  onChange: (v: string) => void
  isDark: boolean
  filter?: (id: string) => boolean
}) {
  const models = useGatewayStore((s) => s.models)
  const opts = models.map((m) => m.id).filter((id) => (filter ? filter(id) : true))

  return (
    <div className="space-y-1">
      <span className="text-zinc-500">Model</span>
      {opts.length > 0 ? (
        <select
          value={opts.includes(value) ? value : ''}
          onChange={(e) => {
            const v = e.target.value
            if (v) onChange(v)
          }}
          className={`w-full px-2 py-1.5 rounded border text-xs font-mono ${
            isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
          }`}
        >
          <option value="">Custom below…</option>
          {opts.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      ) : null}
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-2 py-1.5 rounded border text-[10px] font-mono ${
          isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
        }`}
      />
    </div>
  )
}

export function PropertiesPanel({ isDark }: Props) {
  const nodes = useCanvasStore((s) => s.nodes)
  const selectedNodeIds = useCanvasStore((s) => s.selectedNodeIds)
  const updateNode = useCanvasStore((s) => s.updateNode)
  const removeNode = useCanvasStore((s) => s.removeNode)
  const addConnection = useCanvasStore((s) => s.addConnection)
  const connections = useCanvasStore((s) => s.connections)

  const selectedClipId = useTimelineStore((s) => s.selectedClipId)
  const findClip = useTimelineStore((s) => s.findClip)
  const updateTimelineClip = useTimelineStore((s) => s.updateClip)
  const removeTimelineClip = useTimelineStore((s) => s.removeClip)
  const selectClip = useTimelineStore((s) => s.selectClip)
  const clipSelection = selectedClipId ? findClip(selectedClipId) : null

  const selectedId =
    selectedNodeIds.size === 1
      ? [...selectedNodeIds][0]
      : selectedNodeIds.size >= 1
        ? [...selectedNodeIds][0]
        : null
  const node = selectedId ? nodes.find((n) => n.id === selectedId) : null
  const ids = [...selectedNodeIds]

  const panel = `w-72 shrink-0 flex flex-col border-l min-h-0 overflow-hidden ${
    isDark ? 'bg-zinc-900 border-zinc-800' : 'bg-white border-zinc-200'
  }`

  const isGenKind = (t: string) => t === 'gen-image' || t === 'gen-video'
  const showPrompt =
    node != null &&
    (node.type.startsWith('gen-') || node.type === 'tts' || node.type === 'prompt')

  return (
    <div className={panel}>
      <div
        className={`px-3 py-2 text-xs font-semibold border-b ${
          isDark ? 'border-zinc-800 text-zinc-300' : 'border-zinc-200 text-zinc-800'
        }`}
      >
        Properties
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs">
        {clipSelection && (
          <div
            className={`space-y-2 p-2 rounded-lg border ${isDark ? 'border-blue-500/40 bg-blue-500/10' : 'border-blue-200 bg-blue-50'}`}
          >
            <div className={`font-semibold ${isDark ? 'text-blue-200' : 'text-blue-900'}`}>
              Timeline clip
            </div>
            <div className={`font-mono text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>
              {clipSelection.clip.id}
            </div>
            <div className={isDark ? 'text-zinc-400' : 'text-zinc-700'}>
              {clipSelection.clip.mediaType} · track {clipSelection.trackId}
            </div>
            <label className="block space-y-1">
              <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Label</span>
              <input
                value={clipSelection.clip.label ?? ''}
                onChange={(e) =>
                  updateTimelineClip(clipSelection.trackId, clipSelection.clip.id, {
                    label: e.target.value,
                  })
                }
                className={`w-full px-2 py-1.5 rounded border text-xs ${
                  isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-300'
                }`}
              />
            </label>
            {clipSelection.clip.mediaType === 'image' ? (
              <label className="block space-y-1">
                <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Duration (s)</span>
                <input
                  type="number"
                  step={0.1}
                  min={0.5}
                  value={clipSelection.clip.duration}
                  onChange={(e) => {
                    const v = parseTimelinePositive(e.target.value)
                    if (v == null) return
                    updateTimelineClip(clipSelection.trackId, clipSelection.clip.id, {
                      duration: Math.max(0.5, v),
                    })
                  }}
                  className={`w-full px-2 py-1.5 rounded border text-xs ${
                    isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                  }`}
                />
              </label>
            ) : (
              <>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Source duration (s)</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0.1}
                    value={clipSelection.clip.duration}
                    onChange={(e) => {
                      const v = parseTimelinePositive(e.target.value)
                      if (v == null) return
                      const duration = Math.max(0.1, v)
                      const trims = clampTrimsToDuration(
                        duration,
                        clipSelection.clip.trimStart,
                        clipSelection.clip.trimEnd,
                      )
                      updateTimelineClip(clipSelection.trackId, clipSelection.clip.id, {
                        duration,
                        ...trims,
                      })
                    }}
                    className={`w-full px-2 py-1.5 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Trim start</span>
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      value={clipSelection.clip.trimStart}
                      onChange={(e) => {
                        const v = parseTimelinePositive(e.target.value)
                        if (v == null) return
                        const trims = clampTrimsToDuration(
                          clipSelection.clip.duration,
                          Math.max(0, v),
                          clipSelection.clip.trimEnd,
                        )
                        updateTimelineClip(clipSelection.trackId, clipSelection.clip.id, trims)
                      }}
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Trim end</span>
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      value={clipSelection.clip.trimEnd}
                      onChange={(e) => {
                        const v = parseTimelinePositive(e.target.value)
                        if (v == null) return
                        const trims = clampTrimsToDuration(
                          clipSelection.clip.duration,
                          clipSelection.clip.trimStart,
                          Math.max(0, v),
                        )
                        updateTimelineClip(clipSelection.trackId, clipSelection.clip.id, trims)
                      }}
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                </div>
              </>
            )}
            <label className="block space-y-1">
              <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Source URL</span>
              <textarea
                readOnly
                value={clipSelection.clip.src}
                rows={2}
                className={`w-full px-2 py-1.5 rounded border text-[10px] font-mono resize-none opacity-90 ${
                  isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-200' : 'bg-zinc-100 border-zinc-300'
                }`}
              />
            </label>
            <button
              type="button"
              onClick={() => {
                removeTimelineClip(clipSelection.trackId, clipSelection.clip.id)
                selectClip(null)
              }}
              className="w-full py-2 rounded-lg bg-red-600/20 text-red-400 text-xs"
            >
              Remove from timeline
            </button>
          </div>
        )}
        {ids.length === 2 && (
          <button
            type="button"
            className={`w-full flex items-center justify-center gap-2 py-2 rounded-lg ${
              isDark ? 'bg-blue-600/20 text-blue-300' : 'bg-blue-50 text-blue-800'
            }`}
            onClick={() => {
              const [a, b] = ids
              const exists = connections.some(
                (c) => (c.from === a && c.to === b) || (c.from === b && c.to === a),
              )
              if (!exists) addConnection({ from: a, to: b })
            }}
          >
            <Link2 className="size-3.5" />
            Connect two selected
          </button>
        )}
        {node ? (
          <>
            <div className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
              <div className="font-mono text-[10px] opacity-70">{node.id}</div>
              <div className="mt-1">{node.type}</div>
            </div>
            <label className="block space-y-1">
              <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Title</span>
              <input
                value={node.data.title ?? ''}
                onChange={(e) => updateNode(node.id, { data: { ...node.data, title: e.target.value } })}
                className={`w-full px-2 py-1.5 rounded border text-xs ${
                  isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-300'
                }`}
              />
            </label>

            {node.type === 'gen-image' && (
              <>
                <ModelPicker
                  value={node.data.model ?? DEFAULT_IMAGE_MODEL}
                  onChange={(v) => updateNode(node.id, { data: { ...node.data, model: v } })}
                  isDark={isDark}
                  filter={(id) =>
                    /flux|kontext|qwen|image|mflux|dall/i.test(id) || id.includes('black-forest')
                  }
                />
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={Boolean(node.data.useImageEdit)}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, useImageEdit: e.target.checked } })
                    }
                    className="rounded"
                  />
                  <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                    Use /v1/images/edits (needs incoming image nodes)
                  </span>
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Size</span>
                  <select
                    value={node.data.imageSize ?? '1024x1024'}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, imageSize: e.target.value } })
                    }
                    className={`w-full px-2 py-1.5 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  >
                    {IMAGE_SIZES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}

            {node.type === 'gen-video' && (
              <>
                <ModelPicker
                  value={node.data.model ?? DEFAULT_VIDEO_MODEL}
                  onChange={(v) => updateNode(node.id, { data: { ...node.data, model: v } })}
                  isDark={isDark}
                  filter={(id) => /ltx|video|wan|lightricks/i.test(id) || id.includes('prince-canuma')}
                />
                <div className="grid grid-cols-2 gap-2">
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Width</span>
                    <input
                      type="number"
                      step={32}
                      value={node.data.videoWidth ?? 512}
                      onChange={(e) =>
                        updateNode(node.id, {
                          data: { ...node.data, videoWidth: Number(e.target.value) },
                        })
                      }
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Height</span>
                    <input
                      type="number"
                      step={32}
                      value={node.data.videoHeight ?? 512}
                      onChange={(e) =>
                        updateNode(node.id, {
                          data: { ...node.data, videoHeight: Number(e.target.value) },
                        })
                      }
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                </div>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>num_frames</span>
                  <select
                    value={String(node.data.numFrames ?? 97)}
                    onChange={(e) =>
                      updateNode(node.id, {
                        data: { ...node.data, numFrames: Number(e.target.value) },
                      })
                    }
                    className={`w-full px-2 py-1.5 rounded border text-xs font-mono ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  >
                    {VIDEO_NUM_FRAMES_OPTIONS.map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>fps</span>
                    <input
                      type="number"
                      value={node.data.fps ?? 24}
                      onChange={(e) =>
                        updateNode(node.id, { data: { ...node.data, fps: Number(e.target.value) } })
                      }
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                  <label className="block space-y-1">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>cfg_scale</span>
                    <input
                      type="number"
                      step={0.5}
                      value={node.data.cfgScale ?? 3}
                      onChange={(e) =>
                        updateNode(node.id, {
                          data: { ...node.data, cfgScale: Number(e.target.value) },
                        })
                      }
                      className={`w-full px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    />
                  </label>
                </div>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>pipeline</span>
                  <select
                    value={node.data.pipeline ?? 'distilled'}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, pipeline: e.target.value } })
                    }
                    className={`w-full px-2 py-1.5 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  >
                    {VIDEO_PIPELINES.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>audio_cfg_scale</span>
                  <input
                    type="number"
                    step={0.5}
                    value={node.data.audioCfgScale ?? 7}
                    onChange={(e) =>
                      updateNode(node.id, {
                        data: { ...node.data, audioCfgScale: Number(e.target.value) },
                      })
                    }
                    className={`w-full px-2 py-1 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>image_strength</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={1}
                    value={node.data.imageStrength ?? 1}
                    onChange={(e) =>
                      updateNode(node.id, {
                        data: { ...node.data, imageStrength: Number(e.target.value) },
                      })
                    }
                    className={`w-full px-2 py-1 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>tiling</span>
                  <select
                    value={node.data.tiling ?? 'auto'}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, tiling: e.target.value } })
                    }
                    className={`w-full px-2 py-1.5 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  >
                    {['auto', 'none', 'conservative', 'aggressive', 'spatial', 'temporal'].map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>
                    text_encoder_repo (empty = default)
                  </span>
                  <input
                    value={node.data.textEncoderRepo ?? ''}
                    onChange={(e) =>
                      updateNode(node.id, {
                        data: {
                          ...node.data,
                          textEncoderRepo: e.target.value || undefined,
                        },
                      })
                    }
                    placeholder="mlx-community/gemma-3-12b-it-bf16"
                    className={`w-full px-2 py-1 rounded border text-[10px] font-mono ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
              </>
            )}

            {node.type === 'chat' && (
              <>
                <ModelPicker
                  value={node.data.model ?? DEFAULT_CHAT_MODEL}
                  onChange={(v) => updateNode(node.id, { data: { ...node.data, model: v } })}
                  isDark={isDark}
                  filter={(id) =>
                    /instruct|qwen|llama|gemma|phi|mistral|gguf/i.test(id) ||
                    id.includes('mlx-community')
                  }
                />
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>temperature</span>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    max={2}
                    value={node.data.temperature ?? 0.7}
                    onChange={(e) =>
                      updateNode(node.id, {
                        data: { ...node.data, temperature: Number(e.target.value) },
                      })
                    }
                    className={`w-full px-2 py-1 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
              </>
            )}

            {node.type === 'tts' && (
              <>
                <ModelPicker
                  value={node.data.model ?? DEFAULT_TTS_MODEL}
                  onChange={(v) => updateNode(node.id, { data: { ...node.data, model: v } })}
                  isDark={isDark}
                  filter={(id) => /fish|tts|audio|speech/i.test(id)}
                />
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>voice</span>
                  <input
                    value={node.data.voice ?? 'af_sky'}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, voice: e.target.value } })
                    }
                    className={`w-full px-2 py-1 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>format</span>
                  <select
                    value={node.data.ttsFormat ?? 'wav'}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, ttsFormat: e.target.value } })
                    }
                    className={`w-full px-2 py-1.5 rounded border text-xs ${
                      isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                    }`}
                  >
                    <option value="wav">wav</option>
                    <option value="mp3">mp3</option>
                    <option value="ogg">ogg</option>
                  </select>
                </label>
              </>
            )}

            {node.type === 'stt' && (
              <ModelPicker
                value={node.data.sttModel ?? DEFAULT_STT_MODEL}
                onChange={(v) => updateNode(node.id, { data: { ...node.data, sttModel: v } })}
                isDark={isDark}
                filter={(id) => /whisper|stt|transcrib|paraformer/i.test(id)}
              />
            )}

            {showPrompt && (
              <label className="block space-y-1">
                <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Prompt / text</span>
                <textarea
                  value={node.data.prompt ?? ''}
                  onChange={(e) =>
                    updateNode(node.id, { data: { ...node.data, prompt: e.target.value } })
                  }
                  rows={4}
                  className={`w-full px-2 py-1.5 rounded border text-xs resize-none ${
                    isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-300'
                  }`}
                />
              </label>
            )}

            {(node.type === 'media' || isGenKind(node.type)) && (
              <>
                <label className="block space-y-1">
                  <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Media URL</span>
                  <textarea
                    value={node.data.content ?? ''}
                    onChange={(e) =>
                      updateNode(node.id, { data: { ...node.data, content: e.target.value } })
                    }
                    rows={3}
                    className={`w-full px-2 py-1.5 rounded border text-xs resize-none font-mono ${
                      isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-300'
                    }`}
                  />
                </label>
                {(node.type === 'media' || node.type === 'gen-image') &&
                  node.data.previewType !== 'video' && (
                    <p className={`text-[10px] ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>
                      Double-click the image on the canvas to crop, resize, and compress.
                    </p>
                  )}
                {(node.type === 'media' || node.type === 'gen-image') && (
                  <label className="flex items-center gap-2">
                    <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Preview</span>
                    <select
                      value={node.data.previewType ?? 'image'}
                      onChange={(e) =>
                        updateNode(node.id, {
                          data: {
                            ...node.data,
                            previewType: e.target.value as 'image' | 'video',
                          },
                        })
                      }
                      className={`flex-1 px-2 py-1 rounded border text-xs ${
                        isDark ? 'bg-zinc-800 border-zinc-700' : 'bg-white border-zinc-300'
                      }`}
                    >
                      <option value="image">image</option>
                      <option value="video">video</option>
                    </select>
                  </label>
                )}
              </>
            )}

            <button
              type="button"
              onClick={() => removeNode(node.id)}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-red-600/20 text-red-400"
            >
              <Trash2 className="size-3.5" />
              Delete node
            </button>
          </>
        ) : (
          <p className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>
            Select a node. Ctrl/Cmd+click two nodes, then Connect.
          </p>
        )}
      </div>
    </div>
  )
}
