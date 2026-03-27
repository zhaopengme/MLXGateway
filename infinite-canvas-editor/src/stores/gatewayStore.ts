import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface GatewayModelInfo {
  owned_by?: string
  permission?: unknown[]
  id: string
}

interface GatewayState {
  baseUrl: string
  apiKey: string
  connected: boolean
  lastHealthError: string | null
  models: GatewayModelInfo[]
  modelsFetchedAt: number | null

  setBaseUrl: (u: string) => void
  setApiKey: (k: string) => void
  setConnected: (v: boolean) => void
  setModels: (m: GatewayModelInfo[]) => void
  setLastHealthError: (e: string | null) => void

  /** GET /health — sets connected + lastHealthError */
  checkHealth: () => Promise<boolean>
  /** GET /v1/models */
  fetchModels: () => Promise<void>
}

function normalizeBaseUrl(url: string): string {
  const t = url.trim().replace(/\/$/, '')
  return t || 'http://localhost:8008'
}

export const useGatewayStore = create(
  persist<GatewayState>(
    (set, get) => ({
      baseUrl: 'http://localhost:8008',
      apiKey: '',
      connected: false,
      lastHealthError: null,
      models: [],
      modelsFetchedAt: null,

      setBaseUrl: (u) => set({ baseUrl: normalizeBaseUrl(u) }),
      setApiKey: (apiKey) => set({ apiKey }),
      setConnected: (connected) => set({ connected }),
      setModels: (models) => set({ models, modelsFetchedAt: Date.now() }),
      setLastHealthError: (lastHealthError) => set({ lastHealthError }),

      checkHealth: async () => {
        const base = get().baseUrl
        try {
          const res = await fetch(`${normalizeBaseUrl(base)}/health`, {
            method: 'GET',
          })
          if (!res.ok) {
            const msg = `HTTP ${res.status}`
            set({ connected: false, lastHealthError: msg })
            return false
          }
          const j = (await res.json()) as { status?: string }
          if (j.status === 'ok') {
            set({ connected: true, lastHealthError: null })
            return true
          }
          set({ connected: false, lastHealthError: 'Unexpected health response' })
          return false
        } catch (e) {
          const msg = e instanceof Error ? e.message : 'Network error'
          set({ connected: false, lastHealthError: msg })
          return false
        }
      },

      fetchModels: async () => {
        const { baseUrl, apiKey } = get()
        const base = normalizeBaseUrl(baseUrl)
        const headers: Record<string, string> = {}
        if (apiKey) headers.Authorization = `Bearer ${apiKey}`
        const res = await fetch(`${base}/v1/models`, { headers })
        if (!res.ok) {
          set({ models: [] })
          return
        }
        const j = (await res.json()) as { data?: GatewayModelInfo[] }
        const list = Array.isArray(j.data) ? j.data : []
        set({ models: list, modelsFetchedAt: Date.now() })
      },
    }),
    { name: 'infinite-canvas-gateway' },
  ),
)

export function getGatewayBaseUrl(): string {
  const raw = useGatewayStore.getState().baseUrl
  return normalizeBaseUrl(raw)
}

export function getGatewayHeaders(json = true): HeadersInit {
  const { apiKey } = useGatewayStore.getState()
  const h: Record<string, string> = {}
  if (json) h['Content-Type'] = 'application/json'
  if (apiKey) h.Authorization = `Bearer ${apiKey}`
  return h
}
