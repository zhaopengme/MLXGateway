export async function videoFirstFrameDataUrl(
  src: string,
  maxW = 160,
  maxH = 90,
): Promise<string | undefined> {
  return new Promise((resolve) => {
    const video = document.createElement('video')
    video.crossOrigin = 'anonymous'
    video.muted = true
    video.playsInline = true
    video.preload = 'metadata'

    const cleanup = () => {
      video.removeAttribute('src')
      video.load()
    }

    video.onloadeddata = () => {
      try {
        video.currentTime = Math.min(0.1, (video.duration || 1) * 0.01)
      } catch {
        cleanup()
        resolve(undefined)
      }
    }

    video.onseeked = () => {
      try {
        const vw = video.videoWidth || 1
        const vh = video.videoHeight || 1
        const scale = Math.min(maxW / vw, maxH / vh, 1)
        const cw = Math.round(vw * scale)
        const ch = Math.round(vh * scale)
        const c = document.createElement('canvas')
        c.width = cw
        c.height = ch
        const ctx = c.getContext('2d')
        if (!ctx) {
          cleanup()
          resolve(undefined)
          return
        }
        ctx.drawImage(video, 0, 0, cw, ch)
        const url = c.toDataURL('image/jpeg', 0.85)
        cleanup()
        resolve(url)
      } catch {
        cleanup()
        resolve(undefined)
      }
    }

    video.onerror = () => {
      cleanup()
      resolve(undefined)
    }

    video.src = src
  })
}

export async function imageScaledDataUrl(
  src: string,
  maxW = 160,
  maxH = 90,
): Promise<string | undefined> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      try {
        const iw = img.naturalWidth || 1
        const ih = img.naturalHeight || 1
        const scale = Math.min(maxW / iw, maxH / ih, 1)
        const cw = Math.round(iw * scale)
        const ch = Math.round(ih * scale)
        const c = document.createElement('canvas')
        c.width = cw
        c.height = ch
        const ctx = c.getContext('2d')
        if (!ctx) {
          resolve(undefined)
          return
        }
        ctx.drawImage(img, 0, 0, cw, ch)
        resolve(c.toDataURL('image/jpeg', 0.85))
      } catch {
        resolve(undefined)
      }
    }
    img.onerror = () => resolve(undefined)
    img.src = src
  })
}
