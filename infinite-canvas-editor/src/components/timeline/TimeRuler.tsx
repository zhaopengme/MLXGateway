type Props = {
  totalSec: number
  pxPerSec: number
  isDark: boolean
}

export function TimeRuler({ totalSec, pxPerSec, isDark }: Props) {
  const width = Math.max(400, totalSec * pxPerSec + 80)
  let step = 1
  if (pxPerSec < 12) step = 10
  else if (pxPerSec < 25) step = 5
  else if (pxPerSec < 50) step = 2

  const lastSec = Math.max(0, Math.ceil(totalSec) + 2)
  const ticks: number[] = []
  for (let s = 0; s <= lastSec; s += step) ticks.push(s)

  const line = isDark ? 'bg-zinc-600' : 'bg-zinc-400'
  const label = isDark ? 'text-zinc-500' : 'text-zinc-400'

  return (
    <div
      className={`relative h-6 shrink-0 border-b ${isDark ? 'border-zinc-800 bg-zinc-950/80' : 'border-zinc-200 bg-zinc-100/80'}`}
      style={{ width }}
    >
      {ticks.map((sec) => (
        <div
          key={sec}
          className="absolute top-0 flex flex-col items-start"
          style={{ left: sec * pxPerSec, transform: 'translateX(0)' }}
        >
          <div className={`w-px h-2 shrink-0 ${line}`} />
          <span className={`text-[9px] tabular-nums pl-0.5 ${label}`}>{sec}s</span>
        </div>
      ))}
    </div>
  )
}
