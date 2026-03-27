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
  '<cropper-image crossorigin="anonymous" scalable></cropper-image>' +
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
  const [cropReadout, setCropReadout] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [resizeW, setResizeW] = useState(0)
  const [resizeH, setResizeH] = useState(0)
  const [resizeLockRatio, setResizeLockRatio] = useState(false)
  const [outputQuality, setOutputQuality] = useState(0.85)
  const [outputFormat, setOutputFormat] = useState<ImageProcessOutputFormat>('image/jpeg')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const previewRevokeRef = useRef<string | null>(null)
  const [sidebarTab, setSidebarTab] = useState<'crop' | 'resize'>('crop')

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
    setCropReadout({ x: sel.x, y: sel.y, w: sel.width, h: sel.height })
  }

  useEffect(() => {
    if (!open) return
    setResizeW(0)
    setResizeH(0)
    setResizeLockRatio(false)
    setOutputQuality(0.85)
    setOutputFormat('image/jpeg')
    setErr(null)
    revokePreview()
    setSidebarTab('crop')
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
          .catch(() => {})

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

  /** Crop tab: show selection + shade mask. Resize tab: hide both for a clean preview of the full image. */
  useEffect(() => {
    if (!open || naturalW == null) return
    const cropper = cropperRef.current
    const canvasEl = cropper?.getCropperCanvas() ?? null
    const sel = cropper?.getCropperSelection() ?? null
    if (!canvasEl || !sel) return

    if (sidebarTab === 'crop') {
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
    } else {
      canvasEl.querySelector('cropper-shade')?.remove()
      sel.hidden = true
    }
  }, [open, sidebarTab, naturalW])

  useEffect(() => {
    if (!open || naturalW == null || sidebarTab !== 'resize') {
      revokePreview()
      return
    }
    const cropper = cropperRef.current
    const sel = cropper?.getCropperSelection()
    if (!sel || cropReadout.w <= 0 || cropReadout.h <= 0) {
      revokePreview()
      return
    }

    const rw = resizeW > 0 ? resizeW : 0
    const rh = resizeH > 0 ? resizeH : 0
    let toOpts: { width?: number; height?: number } = {}
    if (rw > 0 && rh > 0) toOpts = { width: rw, height: rh }
    else if (rw > 0) toOpts = { width: rw }
    else if (rh > 0) toOpts = { height: rh }

    let stale = false
    const id = window.setTimeout(() => {
      void (async () => {
        try {
          const canvas =
            Object.keys(toOpts).length > 0 ? await sel.$toCanvas(toOpts) : await sel.$toCanvas()
          if (stale) return
          const blob = await canvasToBlob(canvas, outputFormat, outputQuality)
          if (stale) return
          const url = URL.createObjectURL(blob)
          if (previewRevokeRef.current) URL.revokeObjectURL(previewRevokeRef.current)
          previewRevokeRef.current = url
          setPreviewUrl(url)
        } catch {
          if (!stale) revokePreview()
        }
      })()
    }, 180)

    return () => {
      stale = true
      clearTimeout(id)
    }
  }, [
    open,
    naturalW,
    resizeW,
    resizeH,
    outputFormat,
    outputQuality,
    cropReadout.x,
    cropReadout.y,
    cropReadout.w,
    cropReadout.h,
    sidebarTab,
    revokePreview,
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
    if (!nodeId) return
    const sel = cropperRef.current?.getCropperSelection()
    if (!sel) {
      setErr('Cropper not ready')
      return
    }
    setBusy(true)
    setErr(null)
    try {
      const rw = resizeW > 0 ? resizeW : 0
      const rh = resizeH > 0 ? resizeH : 0
      let toOpts: { width?: number; height?: number } = {}
      if (rw > 0 && rh > 0) toOpts = { width: rw, height: rh }
      else if (rw > 0) toOpts = { width: rw }
      else if (rh > 0) toOpts = { height: rh }

      const canvas =
        Object.keys(toOpts).length > 0 ? await sel.$toCanvas(toOpts) : await sel.$toCanvas()
      const blob = await canvasToBlob(canvas, outputFormat, outputQuality)

      const n = useCanvasStore.getState().nodes.find((x) => x.id === nodeId)
      const d = n?.data
      if (!d) {
        close()
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

  const cw = cropReadout.w
  const ch = cropReadout.h
  const outW = resizeW > 0 ? resizeW : cw
  const outH = resizeH > 0 ? resizeH : ch

  const panelClass = isDark ? 'bg-zinc-900 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-200 text-zinc-900'
  const subBtn =
    isDark ? 'bg-zinc-700 text-zinc-200 hover:bg-zinc-600' : 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300'
  const btnPrimary = `px-3 py-1.5 rounded text-xs font-medium flex items-center gap-1.5 justify-center`
  const tabActive = 'bg-violet-600 text-white hover:bg-violet-700'
  const tabIdle = subBtn

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
              (double-click on canvas · scroll to zoom image)
            </span>
          </h2>
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
            <div
              className={`flex rounded overflow-hidden border p-0.5 gap-0.5 ${
                isDark ? 'border-zinc-600' : 'border-zinc-300'
              }`}
            >
              <button
                type="button"
                role="tab"
                aria-selected={sidebarTab === 'crop'}
                onClick={() => setSidebarTab('crop')}
                className={`flex-1 py-1 rounded text-[11px] font-medium ${sidebarTab === 'crop' ? tabActive : tabIdle}`}
              >
                Crop
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={sidebarTab === 'resize'}
                onClick={() => setSidebarTab('resize')}
                className={`flex-1 py-1 rounded text-[11px] font-medium ${sidebarTab === 'resize' ? tabActive : tabIdle}`}
              >
                Resize
              </button>
            </div>

            {sidebarTab === 'crop' && (
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
                    Crop (canvas): {Math.round(cropReadout.x)}, {Math.round(cropReadout.y)}
                  </div>
                  <div>
                    Size: {Math.round(cw)}×{Math.round(ch)}
                  </div>
                </div>
              </div>
            )}

            {sidebarTab === 'resize' && (
              <div className="flex flex-col gap-2 min-h-0">
                <label className="block space-y-0.5">
                  <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Width (0 = keep crop size)</span>
                  <input
                    type="number"
                    min={0}
                    className={`w-full px-2 py-1 rounded border text-xs ${isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'}`}
                    value={resizeW || ''}
                    onChange={(e) => {
                      const v = Math.max(0, Math.round(parseFloat(e.target.value) || 0))
                      setResizeW(v)
                      if (resizeLockRatio && v > 0 && cw > 0 && ch > 0) {
                        setResizeH(Math.max(1, Math.round((ch / cw) * v)))
                      }
                    }}
                  />
                </label>
                <label className="block space-y-0.5">
                  <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Height (0 = keep crop size)</span>
                  <input
                    type="number"
                    min={0}
                    className={`w-full px-2 py-1 rounded border text-xs ${isDark ? 'bg-zinc-800 border-zinc-600' : 'bg-white border-zinc-300'}`}
                    value={resizeH || ''}
                    onChange={(e) => {
                      const v = Math.max(0, Math.round(parseFloat(e.target.value) || 0))
                      setResizeH(v)
                      if (resizeLockRatio && v > 0 && cw > 0 && ch > 0) {
                        setResizeW(Math.max(1, Math.round((cw / ch) * v)))
                      }
                    }}
                  />
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-[11px]">
                  <input
                    type="checkbox"
                    checked={resizeLockRatio}
                    onChange={(e) => setResizeLockRatio(e.target.checked)}
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
                        Change size or switch to Crop…
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <label className="block space-y-0.5">
              <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
                Quality ({outputQuality.toFixed(2)})
              </span>
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={outputQuality}
                onChange={(e) => setOutputQuality(Number(e.target.value))}
                className="w-full"
              />
            </label>
            <label className="block space-y-0.5">
              <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>Format</span>
              <select
                value={outputFormat}
                onChange={(e) => setOutputFormat(e.target.value as ImageProcessOutputFormat)}
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
              {naturalW != null && naturalH != null ? `${naturalW}×${naturalH}` : '—'} → crop{' '}
              {cw > 0 && ch > 0 ? `${Math.round(cw)}×${Math.round(ch)}` : '—'} → out{' '}
              {outW > 0 && outH > 0 ? `${Math.round(outW)}×${Math.round(outH)}` : '—'}
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
