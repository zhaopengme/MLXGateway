import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'
import type { TimelineClip } from '../types/timeline'

const CORE_VERSION = '0.12.10'
const CORE_BASE = `https://unpkg.com/@ffmpeg/core@${CORE_VERSION}/dist/esm`

let ffmpegSingleton: FFmpeg | null = null
let loadPromise: Promise<FFmpeg> | null = null

export async function getFFmpeg(onLog?: (msg: string) => void): Promise<FFmpeg> {
  if (ffmpegSingleton?.loaded) return ffmpegSingleton
  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    const ffmpeg = new FFmpeg()
    if (onLog) ffmpeg.on('log', ({ message }) => onLog(message))
    await ffmpeg.load({
      coreURL: await toBlobURL(`${CORE_BASE}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${CORE_BASE}/ffmpeg-core.wasm`, 'application/wasm'),
    })
    ffmpegSingleton = ffmpeg
    return ffmpeg
  })()

  return loadPromise
}

function guessImageExt(src: string): string {
  const lower = src.split('?')[0].toLowerCase()
  if (lower.endsWith('.png')) return 'png'
  if (lower.endsWith('.webp')) return 'webp'
  if (lower.endsWith('.gif')) return 'gif'
  return 'jpg'
}

function guessAudioExt(src: string): string {
  const lower = src.split('?')[0].toLowerCase()
  if (lower.endsWith('.wav')) return 'wav'
  if (lower.endsWith('.ogg')) return 'ogg'
  if (lower.endsWith('.m4a')) return 'm4a'
  if (lower.endsWith('.aac')) return 'aac'
  if (lower.endsWith('.flac')) return 'flac'
  return 'mp3'
}

/** Build segment files `seg0.mp4`, ... concat to out.mp4. Optional `audioClips` muxed if non-empty. */
export async function exportClipsToMp4(
  clips: TimelineClip[],
  audioClips?: TimelineClip[],
  onProgress?: (p: number, label: string) => void,
): Promise<Blob> {
  if (!clips.length) throw new Error('No clips to export')

  const ffmpeg = await getFFmpeg()
  const total = clips.length

  for (let i = 0; i < clips.length; i++) {
    const clip = clips[i]
    const pct = Math.round(((i + 0.3) / Math.max(total + (audioClips?.length ? 1 : 0), 1)) * 90)
    onProgress?.(pct, `Processing clip ${i + 1}/${total}`)

    const raw = `in_${i}`
    if (clip.mediaType === 'image') {
      const ext = guessImageExt(clip.src)
      await ffmpeg.writeFile(`${raw}.${ext}`, await fetchFile(clip.src))
      const dur = Math.max(0.5, clip.duration)
      await ffmpeg.exec([
        '-loop',
        '1',
        '-i',
        `${raw}.${ext}`,
        '-c:v',
        'libx264',
        '-t',
        String(dur),
        '-pix_fmt',
        'yuv420p',
        '-vf',
        'scale=1280:-2',
        `seg${i}.mp4`,
      ])
    } else {
      const ext = clip.src.split('?')[0].toLowerCase().endsWith('.webm') ? 'webm' : 'mp4'
      await ffmpeg.writeFile(`${raw}.${ext}`, await fetchFile(clip.src))
      const args = ['-i', `${raw}.${ext}`]
      if (clip.trimStart > 0) {
        args.push('-ss', String(clip.trimStart))
      }
      const playDur = Math.max(0.1, clip.duration - clip.trimStart - clip.trimEnd)
      args.push(
        '-t',
        String(playDur),
        '-c:v',
        'libx264',
        '-an',
        '-pix_fmt',
        'yuv420p',
        `seg${i}.mp4`,
      )
      await ffmpeg.exec(args)
    }
  }

  onProgress?.(88, 'Concatenating video')

  const list = clips.map((_, i) => `file 'seg${i}.mp4'`).join('\n')
  await ffmpeg.writeFile('list.txt', new TextEncoder().encode(list))
  await ffmpeg.exec(['-f', 'concat', '-safe', '0', '-i', 'list.txt', '-c', 'copy', 'video_only.mp4'])

  let readName = 'video_only.mp4'

  if (audioClips?.length) {
    onProgress?.(90, 'Mixing audio track')
    for (let j = 0; j < audioClips.length; j++) {
      const ac = audioClips[j]
      const ext = guessAudioExt(ac.src)
      await ffmpeg.writeFile(`ain_${j}.${ext}`, await fetchFile(ac.src))
      const playDur = Math.max(0.1, ac.duration - ac.trimStart - ac.trimEnd)
      const aargs = ['-i', `ain_${j}.${ext}`]
      if (ac.trimStart > 0) aargs.push('-ss', String(ac.trimStart))
      aargs.push('-t', String(playDur), '-c:a', 'aac', '-b:a', '128k', `aseg${j}.m4a`)
      await ffmpeg.exec(aargs)
    }
    const alist = audioClips.map((_, j) => `file 'aseg${j}.m4a'`).join('\n')
    await ffmpeg.writeFile('alist.txt', new TextEncoder().encode(alist))
    await ffmpeg.exec([
      '-f',
      'concat',
      '-safe',
      '0',
      '-i',
      'alist.txt',
      '-c',
      'copy',
      'audio_all.m4a',
    ])
    await ffmpeg.exec([
      '-i',
      'video_only.mp4',
      '-i',
      'audio_all.m4a',
      '-c:v',
      'copy',
      '-c:a',
      'aac',
      '-map',
      '0:v',
      '-map',
      '1:a',
      '-shortest',
      'out.mp4',
    ])
    readName = 'out.mp4'
    try {
      await ffmpeg.deleteFile('audio_all.m4a')
    } catch {
      /* ok */
    }
    try {
      await ffmpeg.deleteFile('alist.txt')
    } catch {
      /* ok */
    }
    for (let j = 0; j < audioClips.length; j++) {
      try {
        await ffmpeg.deleteFile(`aseg${j}.m4a`)
      } catch {
        /* ok */
      }
      try {
        const ext = guessAudioExt(audioClips[j].src)
        await ffmpeg.deleteFile(`ain_${j}.${ext}`)
      } catch {
        /* ok */
      }
    }
    try {
      await ffmpeg.deleteFile('video_only.mp4')
    } catch {
      /* ok */
    }
  }

  onProgress?.(100, 'Done')
  const data = (await ffmpeg.readFile(readName)) as Uint8Array
  const blob = new Blob([new Uint8Array(data)], { type: 'video/mp4' })

  for (let i = 0; i < clips.length; i++) {
    try {
      await ffmpeg.deleteFile(`seg${i}.mp4`)
    } catch {
      /* ignore */
    }
    try {
      await ffmpeg.deleteFile(`in_${i}.mp4`)
    } catch {
      /* ignore */
    }
    try {
      await ffmpeg.deleteFile(`in_${i}.webm`)
    } catch {
      /* ignore */
    }
    try {
      const ext = guessImageExt(clips[i].src)
      await ffmpeg.deleteFile(`in_${i}.${ext}`)
    } catch {
      /* ignore */
    }
  }
  try {
    await ffmpeg.deleteFile('list.txt')
  } catch {
    /* ignore */
  }
  try {
    await ffmpeg.deleteFile('out.mp4')
  } catch {
    /* ignore */
  }
  try {
    await ffmpeg.deleteFile('video_only.mp4')
  } catch {
    /* ignore */
  }

  return blob
}
