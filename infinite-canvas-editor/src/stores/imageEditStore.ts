import { create } from 'zustand'

interface ImageEditState {
  open: boolean
  nodeId: string | null
  sourceUrl: string | null
  openEditor: (nodeId: string, sourceUrl: string) => void
  close: () => void
}

export const useImageEditStore = create<ImageEditState>((set) => ({
  open: false,
  nodeId: null,
  sourceUrl: null,
  openEditor: (nodeId, sourceUrl) => set({ open: true, nodeId, sourceUrl }),
  close: () => set({ open: false, nodeId: null, sourceUrl: null }),
}))
