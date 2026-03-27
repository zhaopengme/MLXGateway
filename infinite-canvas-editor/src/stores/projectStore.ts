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
      history: [
        {
          id: 'h-sample-vid',
          type: 'video',
          url: 'https://www.w3schools.com/html/mov_bbb.mp4',
          prompt: 'Sample Big Buck Bunny (demo)',
          time: new Date().toISOString(),
        },
        {
          id: 'h-sample-img',
          type: 'image',
          url: 'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800&q=80',
          prompt: 'Sample landscape',
          time: new Date().toISOString(),
        },
      ],

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
