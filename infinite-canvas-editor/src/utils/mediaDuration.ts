const METADATA_TIMEOUT_MS = 12_000

function withTimeout<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  return new Promise((resolve) => {
    const t = window.setTimeout(() => resolve(fallback), ms)
    p.then(
      (v) => {
        window.clearTimeout(t)
        resolve(v)
      },
      () => {
        window.clearTimeout(t)
        resolve(fallback)
      },
    )
  })
}

/** Resolve duration from audio metadata (blob or remote URL). */
export function getAudioDurationSec(src: string): Promise<number> {
  const inner = new Promise<number>((resolve) => {
    const a = document.createElement('audio')
    a.preload = 'metadata'
    a.src = src
    const done = (sec: number) => {
      a.removeAttribute('src')
      a.load()
      resolve(sec)
    }
    a.onloadedmetadata = () => {
      const d = a.duration
      done(Number.isFinite(d) && d > 0 ? d : 5)
    }
    a.onerror = () => done(5)
  })
  return withTimeout(inner, METADATA_TIMEOUT_MS, 5)
}

/** Resolve duration from video metadata (blob or remote URL). */
export function getVideoDurationSec(src: string): Promise<number> {
  const inner = new Promise<number>((resolve) => {
    const v = document.createElement('video')
    v.preload = 'metadata'
    v.muted = true
    v.src = src
    const done = (sec: number) => {
      v.removeAttribute('src')
      v.load()
      resolve(sec)
    }
    v.onloadedmetadata = () => {
      const d = v.duration
      done(Number.isFinite(d) && d > 0 ? d : 12)
    }
    v.onerror = () => done(12)
  })
  return withTimeout(inner, METADATA_TIMEOUT_MS, 12)
}
