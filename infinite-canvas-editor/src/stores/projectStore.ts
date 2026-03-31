import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface HistoryItem {
  id: string
  type: 'image' | 'video'
  url: string
  prompt?: string
  time: string
  label?: string
}

interface ProjectState {
  theme: 'dark' | 'light'
  history: HistoryItem[]
  setTheme: (t: 'dark' | 'light') => void
  addHistoryItem: (item: Omit<HistoryItem, 'id' | 'time'> & { id?: string }) => void
  clearHistory: () => void
}

export const useProjectStore = create(
  persist<ProjectState>(
    (set) => ({
      theme: 'dark',
      history: [],

      setTheme: (theme) => set({ theme }),

      addHistoryItem: (item) =>
        set((s) => ({
          history: [
            {
              id: item.id ?? crypto.randomUUID(),
              type: item.type,
              url: item.url,
              prompt: item.prompt,
              label: item.label,
              time: new Date().toISOString(),
            },
            ...s.history,
          ],
        })),

      clearHistory: () => set({ history: [] }),
    }),
    { name: 'infinite-canvas-project' },
  ),
)
