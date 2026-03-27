export type ImageProcessOutputFormat = 'image/jpeg' | 'image/png' | 'image/webp'

export async function loadImage(src: string): Promise<HTMLImageElement> {
  const img = new Image()
  img.crossOrigin = 'anonymous'
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve()
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = src
  })
  return img
}

export type ProcessImageOpts = {
  /** Source rectangle in natural image pixels */
  cropX?: number
  cropY?: number
  cropW?: number
  cropH?: number
  resizeW?: number
  resizeH?: number
  quality?: number
  format?: ImageProcessOutputFormat
}

/**
 * Draw a crop (+ optional resize) from an HTMLImageElement and encode as Blob.
 * Omit crop to use the full image. Omit resize (or use 0) to keep crop pixel size.
 */
export async function processImage(
  img: HTMLImageElement,
  opts: ProcessImageOpts,
): Promise<{ blob: Blob; width: number; height: number }> {
  const nw = img.naturalWidth
  const nh = img.naturalHeight
  if (!nw || !nh) throw new Error('Image has no dimensions')

  let sx = opts.cropX ?? 0
  let sy = opts.cropY ?? 0
  let sw = opts.cropW ?? nw
  let sh = opts.cropH ?? nh

  sx = Math.max(0, Math.min(sx, nw - 1))
  sy = Math.max(0, Math.min(sy, nh - 1))
  sw = Math.max(1, Math.min(sw, nw - sx))
  sh = Math.max(1, Math.min(sh, nh - sy))

  const rwIn = opts.resizeW && opts.resizeW > 0 ? opts.resizeW : 0
  const rhIn = opts.resizeH && opts.resizeH > 0 ? opts.resizeH : 0

  let dw = rwIn > 0 ? rwIn : sw
  let dh = rhIn > 0 ? rhIn : sh

  if (rwIn > 0 && rhIn <= 0) {
    dh = Math.max(1, Math.round((sh / sw) * dw))
  } else if (rhIn > 0 && rwIn <= 0) {
    dw = Math.max(1, Math.round((sw / sh) * dh))
  }

  const canvas = document.createElement('canvas')
  canvas.width = dw
  canvas.height = dh
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get 2D context')

  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh)

  const format = opts.format ?? 'image/jpeg'
  const quality = format === 'image/png' ? undefined : (opts.quality ?? 0.85)

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('canvas.toBlob failed'))),
      format,
      quality,
    )
  })

  return { blob, width: dw, height: dh }
}

export function canvasToBlob(
  canvas: HTMLCanvasElement,
  format: ImageProcessOutputFormat,
  quality: number,
): Promise<Blob> {
  const q = format === 'image/png' ? undefined : quality
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('canvas.toBlob failed'))),
      format,
      q,
    )
  })
}
