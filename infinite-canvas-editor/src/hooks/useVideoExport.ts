import { useCallback, useState } from 'react'
import { exportClipsToMp4 } from '../utils/ffmpeg'
import { useTimelineStore } from '../stores/timelineStore'

export function useVideoExport() {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [label, setLabel] = useState('')

  const exportMp4 = useCallback(async () => {
    const { getMainTrackClips, getAudioTrackClips } = useTimelineStore.getState()
    const main = getMainTrackClips()
    const audio = getAudioTrackClips()
    if (!main.length) {
      window.alert('Add at least one video or image on the main track to export.')
      return
    }
    setLoading(true)
    setProgress(0)
    setLabel('Loading FFmpeg…')
    try {
      const blob = await exportClipsToMp4(main, audio, (p, lbl) => {
        setProgress(p)
        setLabel(lbl)
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `export-${Date.now()}.mp4`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      console.error(e)
      window.alert(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setLoading(false)
      setProgress(0)
      setLabel('')
    }
  }, [])

  return { exportMp4, loading, progress, label }
}
