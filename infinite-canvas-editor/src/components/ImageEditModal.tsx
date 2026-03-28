import Cropper from 'cropperjs'
import { Loader2, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useCanvasStore } from '../stores/canvasStore'
import { useImageEditStore } from '../stores/imageEditStore'
import {
  canvasToBlob,
  type ImageProcessOutputFormat,
} from '../utils/imageProcess'

const revokeIfBlob = (u: string | undefined) => {
  if (u?.startsWith('blob:')) URL.revokeObjectURL(u)
}

const IMAGE_EDIT_CROPPER_TEMPLATE =
  '<cropper-canvas background>' +
  '<cropper-image crossorigin="anonymous" scalable translatable></cropper-image>' +
  '<cropper-handle action="select" plain></cropper-handle>' +
  '<cropper-selection movable resizable zoomable>' +
  '<cropper-grid role="grid" bordered covered></cropper-grid>' +
  '<cropper-crosshair centered></cropper-crosshair>' +
  '<cropper-handle action="move" theme-color="rgba(255, 255, 255, 0.35)"></cropper-handle>' +
  '<cropper-handle action="n-resize"></cropper-handle>' +
  '<cropper-handle action="e-resize"></cropper-handle>' +
  '<cropper-handle action="s-resize"></cropper-handle>' +
  '<cropper-handle action="w-resize"></cropper-handle>' +
  '<cropper-handle action="ne-resize"></cropper-handle>' +
  '<cropper-handle action="nw-resize"></cropper-handle>' +
  '<cropper-handle action="se-resize"></cropper-handle>' +
  '<cropper-handle action="sw-resize"></cropper-handle>' +
  '</cropper-selection>' +
  '</cropper-canvas>'

/** Default: selection covers the full visible image (no crop until user adjusts). */
function selectionCoverFullImage(cropper: Cropper) {
  const canvas = cropper.getCropperCanvas()
  const ci = cropper.getCropperImage()
  const sel = cropper.getCropperSelection()
  if (!canvas || !ci || !sel) return

  const cr = canvas.getBoundingClientRect()
  const ir = ci.getBoundingClientRect()
  const x = Math.round(ir.left - cr.left)
  const y = Math.round(ir.top - cr.top)
  const w = Math.max(1, Math.round(ir.width))
  const h = Math.max(1, Math.round(ir.height))
  sel.aspectRatio = NaN
  sel.initialCoverage = NaN
  sel.$change(x, y, w, h, NaN, true)
  const withInitial = sel as unknown as {
    $initialSelection: { x: number; y: number; width: number; height: number }
  }
  withInitial.$initialSelection = { x, y, width: w, height: h }
}

type Props = { isDark: boolean }

export function ImageEditModal({ isDark }: Props) {
  const open = useImageEditStore((s) => s.open)
  const nodeId = useImageEditStore((s) => s.nodeId)
  const sourceUrl = useImageEditStore((s) => s.sourceUrl)
  const close = useImageEditStore((s) => s.close)
  const updateNode = useCanvasStore((s) => s.updateNode)

  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const cropperRef = useRef<Cropper | null>(null)

  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [naturalW, setNaturalW] = useState<number | null>(null)
  const [naturalH, setNaturalH] = useState<number | null>(null)
  
  // 裁剪状态
  const [cropParams, setCropParams] = useState({ x: 0, y: 0, width: 0, height: 0 })
  
  // 调整大小状态
  const [resizeParams, setResizeParams] = useState({ width: 0, height: 0, lockRatio: false })
  
  // 输出参数
  const [outputParams, setOutputParams] = useState({ 
    quality: 0.85, 
    format: 'image/jpeg' as ImageProcessOutputFormat 
  })
  
  // 预览状态
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const previewRevokeRef = useRef<string | null>(null)
  
  // 当前模式
  const [mode, setMode] = useState<'crop' | 'resize'>('crop')

  const revokePreview = useCallback(() => {
    if (previewRevokeRef.current) {
      URL.revokeObjectURL(previewRevokeRef.current)
      previewRevokeRef.current = null
    }
    setPreviewUrl(null)
  }, [])

  const syncReadout = (cropper: Cropper) => {
    const sel = cropper.getCropperSelection()
    if (!sel) return
    setCropParams({ x: sel.x, y: sel.y, width: sel.width, height: sel.height })
  }

  useEffect(() => {
    if (!open) return
    // 重置状态
    setCropParams({ x: 0, y: 0, width: 0, height: 0 })
    setResizeParams({ width: 0, height: 0, lockRatio: false })
    setOutputParams({ quality: 0.85, format: 'image/jpeg' })
    setErr(null)
    revokePreview()
    setMode('crop')
  }, [open, nodeId, sourceUrl, revokePreview])

  useEffect(() => {
    if (!open || !sourceUrl) {
      cropperRef.current?.destroy()
      cropperRef.current = null
      return
    }

    const container = containerRef.current
    const img = imgRef.current
    if (!container || !img) return

    let cancelled = false
    let detachSelectionListener: (() => void) | undefined
    cropperRef.current?.destroy()
    cropperRef.current = null
    setErr(null)
    setNaturalW(null)
    setNaturalH(null)

    img.crossOrigin = 'anonymous'
    img.removeAttribute('src')
    img.src = sourceUrl

    const onLoad = () => {
      if (cancelled || !containerRef.current || !imgRef.current) return
      try {
        const cropper = new Cropper(imgRef.current, {
          container,
          template: IMAGE_EDIT_CROPPER_TEMPLATE,
        })
        cropperRef.current = cropper
        const ci = cropper.getCropperImage()
        void ci
          ?.$ready()
          .then((im: HTMLImageElement) => {
            if (cancelled) return
            setNaturalW(im.naturalWidth)
            setNaturalH(im.naturalHeight)

            ci.$center('contain')
            requestAnimationFrame(() => {
              if (cancelled) return
              selectionCoverFullImage(cropper)
              syncReadout(cropper)
            })
          })
          .catch((error) => {
            if (!cancelled) setErr(`Failed to load image: ${error instanceof Error ? error.message : 'Unknown error'}`)
          })

        const sel = cropper.getCropperSelection()
        const onSelChange = () => syncReadout(cropper)
        sel?.addEventListener('change', onSelChange)
        detachSelectionListener = () => sel?.removeEventListener('change', onSelChange)
        syncReadout(cropper)

        if (cancelled) {
          detachSelectionListener()
          detachSelectionListener = undefined
          cropper.destroy()
          return
        }
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Cropper failed')
      }
    }

    if (img.complete && img.naturalWidth > 0) onLoad()
    else img.onload = () => onLoad()

    return () => {
      cancelled = true
      img.onload = null
      detachSelectionListener?.()
      cropperRef.current?.destroy()
      cropperRef.current = null
    }
  }, [open, sourceUrl, nodeId])

  /** Crop mode: show selection + shade mask. Resize mode: show full image without crop selection. */
  useEffect(() => {
    if (!open || naturalW == null) return
    const cropper = cropperRef.current
    const canvasEl = cropper?.getCropperCanvas() ?? null
    const sel = cropper?.getCropperSelection() ?? null
    if (!canvasEl || !sel) return

    if (mode === 'crop') {
      // 显示裁剪框和遮罩
      let shade = canvasEl.querySelector('cropper-shade')
      if (!shade) {
        shade = document.createElement('cropper-shade')
        const selectionEl = canvasEl.querySelector('cropper-selection')
        if (selectionEl?.parentNode === canvasEl) {
          canvasEl.insertBefore(shade, selectionEl)
        } else {
          canvasEl.appendChild(shade)
        }
        requestAnimationFrame(() => {
          const s = cropperRef.current?.getCropperSelection()
          if (!s || s.width <= 0 || s.height <= 0) return
          s.$change(s.x, s.y, s.width, s.height, s.aspectRatio, true)
        })
      }
      sel.hidden = false
      // 显示网格
      const grid = canvasEl.querySelector('cropper-grid')
      if (grid) {
        ;(grid as HTMLElement).style.display = 'block'
      }
      // 启用裁剪器交互
      ;(canvasEl as HTMLElement).style.pointerEvents = 'auto'
    } else {
      // 隐藏裁剪框、遮罩和网格，显示完整图片
      canvasEl.querySelector('cropper-shade')?.remove()
      sel.hidden = true
      // 隐藏网格
      const grid = canvasEl.querySelector('cropper-grid')
      if (grid) {
        ;(grid as HTMLElement).style.display = 'none'
      }
      // 禁用裁剪器交互，只显示图片
      ;(canvasEl as HTMLElement).style.pointerEvents = 'none'
      // 确保选择区域覆盖整个图片
      if (cropper) selectionCoverFullImage(cropper)
    }
  }, [open, mode, naturalW])

  // 直接处理原图片的调整大小功能
  const resizeImage = async (url: string, width: number, height: number, format: ImageProcessOutputFormat, quality: number) => {
    return new Promise<Blob>((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas')
          const ctx = canvas.getContext('2d')
          if (!ctx) {
            reject(new Error('Canvas context not available'))
            return
          }
          
          // 计算新的尺寸
          let newWidth = width
          let newHeight = height
          
          if (width === 0 && height === 0) {
            // 使用原始尺寸
            newWidth = img.naturalWidth
            newHeight = img.naturalHeight
          } else if (width === 0) {
            // 按高度比例计算宽度
            newWidth = Math.round((img.naturalWidth / img.naturalHeight) * height)
          } else if (height === 0) {
            // 按宽度比例计算高度
            newHeight = Math.round((img.naturalHeight / img.naturalWidth) * width)
          }
          
          canvas.width = newWidth
          canvas.height = newHeight
          ctx.drawImage(img, 0, 0, newWidth, newHeight)
          
          canvasToBlob(canvas, format, quality)
            .then(blob => {
              if (blob) {
                resolve(blob)
              } else {
                reject(new Error('Failed to create blob'))
              }
            })
            .catch(reject)
        } catch (error) {
          reject(error)
        }
      }
      img.onerror = () => reject(new Error('Failed to load image'))
      img.src = url
    })
  }

  useEffect(() => {
    if (!open || naturalW == null || mode !== 'resize') {
      revokePreview()
      return
    }

    const rw = resizeParams.width > 0 ? resizeParams.width : 0
    const rh = resizeParams.height > 0 ? resizeParams.height : 0

    let stale = false
    const id = window.setTimeout(() => {
      void (async () => {
        try {
          if (!sourceUrl) return
          const blob = await resizeImage(sourceUrl, rw, rh, outputParams.format, outputParams.quality)
          if (stale) return
          const url = URL.createObjectURL(blob)
          if (previewRevokeRef.current) URL.revokeObjectURL(previewRevokeRef.current)
          previewRevokeRef.current = url
          setPreviewUrl(url)
        } catch (error) {
          if (!stale) {
            revokePreview()
            setErr(`Preview generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
          }
        }
      })()
    }, 300) // 增加防抖时间，减少频繁计算

    return () => {
      stale = true
      clearTimeout(id)
    }
  }, [
    open,
    naturalW,
    resizeParams.width,
    resizeParams.height,
    outputParams.format,
    outputParams.quality,
    mode,
    revokePreview,
    sourceUrl
  ])

  useEffect(
    () => () => {
      revokePreview()
    },
    [revokePreview],
  )

  const setAspectPreset = (preset: 'free' | '1:1' | '4:3' | '16:9' | '9:16') => {
    const sel = cropperRef.current?.getCropperSelection()
    if (!sel) return
    const ratio =
      preset === 'free'
        ? NaN
        : preset === '1:1'
          ? 1
          : preset === '4:3'
            ? 4 / 3
            : preset === '16:9'
              ? 16 / 9
              : 9 / 16
    sel.aspectRatio = ratio
    sel.$change(sel.x, sel.y, sel.width, sel.height, ratio, true)
    if (cropperRef.current) syncReadout(cropperRef.current)
  }

  const onResetCrop = () => {
    cropperRef.current?.getCropperSelection()?.$reset()
    if (cropperRef.current) syncReadout(cropperRef.current)
  }

  const onApply = async () => {
    if (!nodeId) {
      setErr('No node selected')
      return
    }
    setBusy(true)
    setErr(null)
    try {
      let blob: Blob
      
      if (mode === 'resize') {
        // 直接对原图片进行调整大小
        if (!sourceUrl) {
          setErr('No image source')
          return
        }
        const rw = resizeParams.width > 0 ? resizeParams.width : 0
        const rh = resizeParams.height > 0 ? resizeParams.height : 0
        blob = await resizeImage(sourceUrl, rw, rh, outputParams.format, outputParams.quality)
      } else {
        // 裁剪模式
        const sel = cropperRef.current?.getCropperSelection()
        if (!sel) {
          setErr('Editor not ready')
          return
        }
        if (cropParams.width <= 0 || cropParams.height <= 0) {
          setErr('Invalid crop area')
          return
        }
        
        const rw = resizeParams.width > 0 ? resizeParams.width : 0
        const rh = resizeParams.height > 0 ? resizeParams.height : 0
        let toOpts: { width?: number; height?: number } = {}
        if (rw > 0 && rh > 0) toOpts = { width: rw, height: rh }
        else if (rw > 0) toOpts = { width: rw }
        else if (rh > 0) toOpts = { height: rh }

        const canvas =
          Object.keys(toOpts).length > 0 ? await sel.$toCanvas(toOpts) : await sel.$toCanvas()
        blob = await canvasToBlob(canvas, outputParams.format, outputParams.quality)
      }

      const n = useCanvasStore.getState().nodes.find((x) => x.id === nodeId)
      const d = n?.data
      if (!d) {
        setErr('Node not found')
        return
      }
      revokeIfBlob(d.content)
      const nextUrl = URL.createObjectURL(blob)
      updateNode(nodeId, {
        data: {
          ...d,
          content: nextUrl,
          previewType: 'image',
          error: undefined,
        },
      })
      close()
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Process failed')
    } finally {
      setBusy(false)
    }
  }

  if (!open || !sourceUrl) return null

  const cw = cropParams.width
  const ch = cropParams.height
  let outW, outH
  if (mode === 'resize') {
    // 调整大小模式：使用指定的尺寸或原始尺寸
    outW = resizeParams.width > 0 ? resizeParams.width : naturalW || 0
    outH = resizeParams.height > 0 ? resizeParams.height : naturalH || 0
    // 如果只指定了一个维度，按比例计算另一个维度
    if (resizeParams.width > 0 && resizeParams.height === 0 && naturalW && naturalH) {
      outH = Math.round((naturalH / naturalW) * outW)
    } else if (resizeParams.height > 0 && resizeParams.width === 0 && naturalW && naturalH) {
      outW = Math.round((naturalW / naturalH) * outH)
    }
  } else {
    // 裁剪模式：使用裁剪后的尺寸
    outW = resizeParams.width > 0 ? resizeParams.width : cw
    outH = resizeParams.height > 0 ? resizeParams.height : ch
  }

  const panelClass = isDark ? 'bg-zinc-900 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-200 text-zinc-900'
  const subBtn =
    isDark ? 'bg-zinc-700 text-zinc-200 hover:bg-zinc-600' : 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300'
  const btnPrimary = `px-3 py-1.5 rounded text-xs font-medium flex items-center gap-1.5 justify-center`
  const modeActive = 'bg-violet-600 text-white hover:bg-violet-700'
  const modeIdle = subBtn

  return (
    <>
      <button
        type="button"
        aria-label="Close image editor"
        className="fixed inset-0 z-50 bg-black/50"
        onClick={() => !busy && close()}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="image-edit-title"
        className={`fixed z-[51] left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(96vw,920px)] h-[min(90vh,720px)] flex flex-col rounded-xl border shadow-2xl overflow-hidden ${panelClass}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className={`shrink-0 flex items-center justify-between px-3 py-2 border-b ${
            isDark ? 'border-zinc-700 bg-zinc-800/80' : 'border-zinc-200 bg-zinc-50'
          }`}
        >
          <h2 id="image-edit-title" className="text-sm font-semibold">
            Edit image{' '}
            <span className="text-[10px] font-normal opacity-70">
              (double-click on canvas · drag to pan · scroll to zoom)
            </span>
          </h2>
          <div className="flex items-center gap-2">
            <div
              className={`flex rounded overflow-hidden border p-0.5 gap-0.5 ${
                isDark ? 'border-zinc-600' : 'border-zinc-300'
              }`}
            >
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'crop'}
                onClick={() => setMode('crop')}
                className={`flex-1 py-1 rounded text-[11px] font-medium ${mode === 'crop' ? modeActive : modeIdle}`}
              >
                Crop
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'resize'}
                onClick={() => setMode('resize')}
                className={`flex-1 py-1 rounded text-[11px] font-medium ${mode === 'resize' ? modeActive : modeIdle}`}
              >
                Resize
              </button>
            </div>
            <button
              type="button"
              disabled={busy}
              onClick={() => close()}
              className={`p-1 rounded ${isDark ? 'hover:bg-zinc-700' : 'hover:bg-zinc-200'}`}
              aria-label="Close"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex min-h-0 gap-2 p-2">
          <div
            ref={containerRef}
            className={`image-edit-modal-cropper flex-1 min-w-0 min-h-[280px] rounded border overflow-hidden ${
              isDark ? 'border-zinc-700 bg-zinc-950' : 'border-zinc-200 bg-zinc-100'
            }`}
          >
            <img ref={imgRef} alt="" className="hidden" />
          </div>

          <div className="w-56 shrink-0 flex flex-col gap-2 text-xs overflow-y-auto">
            {mode === 'crop' && (
              <div className="flex flex-col gap-2 min-h-0">
                <div className="flex flex-wrap gap-1">
                  {(['free', '1:1', '4:3', '16:9', '9:16'] as const).map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setAspectPreset(p)}
                      className={`px-1.5 py-0.5 rounded text-[10px] ${subBtn}`}
                    >
                      {p === 'free' ? 'Free' : p}
                    </button>
                  ))}
                </div>

                <button type="button" onClick={onResetCrop} className={`${btnPrimary} ${subBtn} w-full`}>
                  Reset crop
                </button>

                <div
                  className={`font-mono text-[10px] p-2 rounded border ${
                    isDark ? 'border-zinc-700 bg-zinc-800/40' : 'border-zinc-200 bg-zinc-50'
                  }`}
                >
                  <div>
                    Crop (canvas): {Math.round(cropParams.x)}, {Math.round(cropParams.y)}
                  </div>
                  <div>
                    Size: {Math.round(cw)}×{Math.round(ch)}
                  </div>
                </div>
              </div>
            )}

            {mode === 'resize' && (
              <div className="flex flex-col gap-2 min-h-0">
                <label className="block space-y-0.5">
                  <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Width (0 = keep original size)</span>
                  <input
                    type="number"
                    min={0}
                    className={`w-full px-2 py-1 rounded border text-xs ${isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'}`}
                    value={resizeParams.width || ''}
                    onChange={(e) => {
                      const v = Math.max(0, Math.round(parseFloat(e.target.value) || 0))
                      setResizeParams(prev => {
                        const newParams = { ...prev, width: v }
                        if (prev.lockRatio && v > 0 && naturalW && naturalH) {
                          newParams.height = Math.max(1, Math.round((naturalH / naturalW) * v))
                        }
                        return newParams
                      })
                    }}
                  />
                </label>
                <label className="block space-y-0.5">
                  <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Height (0 = keep original size)</span>
                  <input
                    type="number"
                    min={0}
                    className={`w-full px-2 py-1 rounded border text-xs ${isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'}`}
                    value={resizeParams.height || ''}
                    onChange={(e) => {
                      const v = Math.max(0, Math.round(parseFloat(e.target.value) || 0))
                      setResizeParams(prev => {
                        const newParams = { ...prev, height: v }
                        if (prev.lockRatio && v > 0 && naturalW && naturalH) {
                          newParams.width = Math.max(1, Math.round((naturalW / naturalH) * v))
                        }
                        return newParams
                      })
                    }}
                  />
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-[11px]">
                  <input
                    type="checkbox"
                    checked={resizeParams.lockRatio}
                    onChange={(e) => setResizeParams(prev => ({ ...prev, lockRatio: e.target.checked }))}
                    className="rounded"
                  />
                  Lock ratio
                </label>

                <div className="space-y-1">
                  <div className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Output preview</div>
                  <div
                    className={`rounded border overflow-hidden flex items-center justify-center min-h-[96px] max-h-40 ${
                      isDark ? 'border-zinc-700 bg-zinc-800/40' : 'border-zinc-200 bg-zinc-50'
                    }`}
                  >
                    {previewUrl ? (
                      <img src={previewUrl} alt="" className="max-w-full max-h-36 object-contain" />
                    ) : (
                      <span className="text-[10px] opacity-50 px-2 py-6 text-center">
                        Adjust size to see preview
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <label className="block space-y-0.5">
              <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                Quality ({outputParams.quality.toFixed(2)})
              </span>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={outputParams.quality}
                onChange={(e) => setOutputParams(prev => ({ ...prev, quality: Number(e.target.value) }))}
                className="w-full"
              />
            </label>
            <label className="block space-y-0.5">
              <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Format</span>
              <select
                value={outputParams.format}
                onChange={(e) => setOutputParams(prev => ({ ...prev, format: e.target.value as ImageProcessOutputFormat }))}
                className={`w-full px-2 py-1 rounded border text-xs ${isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'}`}
              >
                <option value="image/jpeg">JPEG</option>
                <option value="image/png">PNG</option>
                <option value="image/webp">WebP</option>
              </select>
            </label>

            <div
              className={`text-[10px] font-mono p-2 rounded border mt-auto ${
                isDark ? 'border-zinc-700 text-zinc-400' : 'border-zinc-200 text-zinc-600'
              }`}
            >
              {mode === 'crop' ? (
                <>{
                  naturalW != null && naturalH != null ? `${naturalW}×${naturalH}` : '—'} → crop{' '}
                  {cw > 0 && ch > 0 ? `${Math.round(cw)}×${Math.round(ch)}` : '—'} → out{' '}
                  {outW > 0 && outH > 0 ? `${Math.round(outW)}×${Math.round(outH)}` : '—'}
                </>
              ) : (
                <>{naturalW != null && naturalH != null ? `${naturalW}×${naturalH}` : '—'} → out{' '}
                  {outW > 0 && outH > 0 ? `${Math.round(outW)}×${Math.round(outH)}` : '—'}
                </>
              )}
            </div>
          </div>
        </div>

        {err && <div className="shrink-0 px-3 py-1 text-[11px] text-red-400">{err}</div>}

        <div
          className={`shrink-0 flex justify-end gap-2 px-3 py-2 border-t ${
            isDark ? 'border-zinc-700 bg-zinc-800/50' : 'border-zinc-200 bg-zinc-50'
          }`}
        >
          <button type="button" disabled={busy} onClick={() => close()} className={`${btnPrimary} ${subBtn}`}>
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => void onApply()}
            className={`${btnPrimary} bg-violet-600 text-white hover:bg-violet-700`}
          >
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : null}
            Apply to node
          </button>
        </div>
      </div>
    </>
  )
}
