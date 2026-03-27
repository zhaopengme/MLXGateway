import { create } from 'zustand'
import type { CanvasNode, Connection, ToolMode, ViewState } from '../types/canvas'
import { useTimelineStore } from './timelineStore'
import { DEFAULT_VIDEO_MODEL } from '../utils/mlxDefaults'

const defaultNodesBase: CanvasNode[] = [
  {
    id: 'n-demo-media',
    type: 'media',
    x: 80,
    y: 120,
    width: 280,
    height: 200,
    data: {
      title: 'Sample image',
      content:
        'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80',
      previewType: 'image',
    },
    settings: {},
  },
  {
    id: 'n-gen-vid',
    type: 'gen-video',
    x: 420,
    y: 100,
    width: 320,
    height: 260,
    data: {
      title: 'Video generation',
      prompt: 'Short cinematic clip, golden hour',
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
  },
  {
    id: 'n-text',
    type: 'text',
    x: 120,
    y: 380,
    width: 240,
    height: 120,
    data: {
      title: 'Note',
      content: 'Connect this to a gen-video or gen-image node for prompt text.',
    },
    settings: {},
  },
]

function cloneDefaultNodes(): CanvasNode[] {
  return JSON.parse(JSON.stringify(defaultNodesBase)) as CanvasNode[]
}

interface CanvasState {
  nodes: CanvasNode[]
  connections: Connection[]
  view: ViewState
  selectedNodeIds: Set<string>
  tool: ToolMode
  dragNodeId: string | null

  setView: (updater: ViewState | ((v: ViewState) => ViewState)) => void
  setTool: (t: ToolMode) => void
  setDragNodeId: (id: string | null) => void

  addNode: (node: Omit<CanvasNode, 'id'> & { id?: string }) => string
  updateNode: (id: string, patch: Partial<CanvasNode>) => void
  removeNode: (id: string) => void

  selectOnly: (id: string | null) => void
  toggleSelect: (id: string) => void
  clearSelection: () => void
  /** Clear selected nodes only (keep timeline clip selection). */
  clearNodeSelectionOnly: () => void
  setSelected: (ids: Set<string>) => void

  addConnection: (c: Omit<Connection, 'id'> & { id?: string }) => string
  removeConnection: (id: string) => void

  resetCanvas: () => void
}

function newId() {
  return crypto.randomUUID()
}

export const useCanvasStore = create<CanvasState>((set) => ({
  nodes: cloneDefaultNodes(),
  connections: [],
  view: { x: 80, y: 60, zoom: 0.85 },
  selectedNodeIds: new Set(),
  tool: 'select',
  dragNodeId: null,

  setView: (updater) =>
    set((s) => ({
      view: typeof updater === 'function' ? updater(s.view) : updater,
    })),

  setTool: (tool) => set({ tool }),
  setDragNodeId: (dragNodeId) => set({ dragNodeId }),

  addNode: (partial) => {
    const id = partial.id ?? newId()
    const node: CanvasNode = {
      id,
      type: partial.type,
      x: partial.x,
      y: partial.y,
      width: partial.width,
      height: partial.height,
      data: partial.data ?? {},
      settings: partial.settings ?? {},
    }
    set((s) => ({ nodes: [...s.nodes, node] }))
    return id
  },

  updateNode: (id, patch) =>
    set((s) => ({
      nodes: s.nodes.map((n) => {
        if (n.id !== id) return n
        const { data: patchData, ...rest } = patch
        return {
          ...n,
          ...rest,
          ...(patchData !== undefined
            ? { data: { ...n.data, ...patchData } }
            : {}),
        }
      }),
    })),

  removeNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      connections: s.connections.filter((c) => c.from !== id && c.to !== id),
      selectedNodeIds: new Set([...s.selectedNodeIds].filter((i) => i !== id)),
    })),

  selectOnly: (id) => {
    useTimelineStore.getState().selectClip(null)
    set({
      selectedNodeIds: id ? new Set([id]) : new Set(),
    })
  },

  toggleSelect: (id) => {
    useTimelineStore.getState().selectClip(null)
    set((s) => {
      const next = new Set(s.selectedNodeIds)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return { selectedNodeIds: next }
    })
  },

  clearSelection: () => {
    useTimelineStore.getState().selectClip(null)
    set({ selectedNodeIds: new Set() })
  },

  clearNodeSelectionOnly: () => set({ selectedNodeIds: new Set() }),

  setSelected: (ids) => {
    useTimelineStore.getState().selectClip(null)
    set({ selectedNodeIds: new Set(ids) })
  },

  addConnection: (partial) => {
    const id = partial.id ?? newId()
    const c: Connection = {
      id,
      from: partial.from,
      to: partial.to,
      fromPort: partial.fromPort,
      toPort: partial.toPort,
    }
    set((s) => ({ connections: [...s.connections, c] }))
    return id
  },

  removeConnection: (id) =>
    set((s) => ({
      connections: s.connections.filter((c) => c.id !== id),
    })),

  resetCanvas: () => {
    useTimelineStore.getState().clearClips()
    set({
      nodes: cloneDefaultNodes(),
      connections: [],
      view: { x: 80, y: 60, zoom: 0.85 },
      selectedNodeIds: new Set(),
    })
  },
}))

export function guessMediaTypeFromNode(n: CanvasNode): 'video' | 'image' | 'audio' | null {
  if (n.type === 'tts' && n.data.audioBlobUrl) return 'audio'
  if (n.data.previewType === 'video') return 'video'
  if (n.data.previewType === 'image') return 'image'
  const url = n.data.content?.split('?')[0].toLowerCase() ?? ''
  if (/\.(mp4|webm|mov|m4v)$/.test(url)) return 'video'
  if (/\.(png|jpe?g|gif|webp|bmp)$/.test(url)) return 'image'
  if (/\.(mp3|wav|ogg|m4a|aac|flac)$/.test(url)) return 'audio'
  if (n.type === 'gen-video') return 'video'
  if (n.type === 'gen-image') return 'image'
  return null
}

export {
  DEFAULT_IMAGE_MODEL,
  DEFAULT_VIDEO_MODEL,
  DEFAULT_CHAT_MODEL,
} from '../utils/mlxDefaults'
