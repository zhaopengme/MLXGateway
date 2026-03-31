import type { CanvasNode } from '../types/canvas'
import { useCanvasStore } from '../stores/canvasStore'

/** Incoming edges → source nodes in connection order */
export function getOrderedIncomingNodes(targetId: string): CanvasNode[] {
  const { nodes, connections } = useCanvasStore.getState()
  const fromIds = connections.filter((c) => c.to === targetId).map((c) => c.from)
  const map = new Map(nodes.map((n) => [n.id, n]))
  return fromIds.map((id) => map.get(id)).filter(Boolean) as CanvasNode[]
}

export function getPromptFromIncoming(incoming: CanvasNode[], nodePrompt: string | undefined): string {
  const trimmed = nodePrompt?.trim()
  if (trimmed) return trimmed
  for (const n of incoming) {
    if (n.type === 'prompt' || n.type === 'text') {
      const t = n.data.content?.trim()
      if (t) return t
    }
  }
  return ''
}

function isHttpOrHttps(url: string): boolean {
  return url.startsWith('http://') || url.startsWith('https://')
}

/** True if node likely provides a raster image URL valid for image_url */
export function isImageLikeNode(n: CanvasNode): boolean {
  const c = n.data.content
  if (!c) return false
  if (n.data.previewType === 'video') return false
  if (n.type === 'gen-image' || n.type === 'gen-image-advanced') return true
  if (n.type === 'media') {
    if (n.data.previewType === 'image') return true
    const base = c.split('?')[0].toLowerCase()
    if (/\.(png|jpe?g|gif|webp|bmp)$/i.test(base)) return true
  }
  return false
}

/** URL suitable for MLXGateway image_url (reachable from gateway host) */
export function getImageUrlFromNode(n: CanvasNode): string | undefined {
  const c = n.data.content
  if (!c) return undefined
  if (isHttpOrHttps(c)) return c
  if (c.startsWith('data:image/')) return c
  return undefined
}

export async function blobUrlToBase64(blobUrl: string): Promise<string> {
  const res = await fetch(blobUrl)
  const buf = await res.arrayBuffer()
  const bytes = new Uint8Array(buf)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!)
  return btoa(binary)
}
