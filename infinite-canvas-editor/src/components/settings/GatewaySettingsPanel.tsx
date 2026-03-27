import { Loader2, RefreshCw, Server } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useGatewayStore } from '../../stores/gatewayStore'

type Props = {
  isDark: boolean
  open: boolean
  onClose: () => void
}

export function GatewaySettingsPanel({ isDark, open, onClose }: Props) {
  const baseUrl = useGatewayStore((s) => s.baseUrl)
  const apiKey = useGatewayStore((s) => s.apiKey)
  const connected = useGatewayStore((s) => s.connected)
  const lastHealthError = useGatewayStore((s) => s.lastHealthError)
  const models = useGatewayStore((s) => s.models)
  const setBaseUrl = useGatewayStore((s) => s.setBaseUrl)
  const setApiKey = useGatewayStore((s) => s.setApiKey)
  const checkHealth = useGatewayStore((s) => s.checkHealth)
  const fetchModels = useGatewayStore((s) => s.fetchModels)

  const [busy, setBusy] = useState(false)
  const [localUrl, setLocalUrl] = useState(baseUrl)
  const [localKey, setLocalKey] = useState(apiKey)

  useEffect(() => {
    if (open) {
      setLocalUrl(baseUrl)
      setLocalKey(apiKey)
    }
  }, [open, baseUrl, apiKey])

  if (!open) return null

  const card = `absolute right-4 top-14 z-50 w-[min(100vw-2rem,380px)] rounded-xl border shadow-xl p-4 ${
    isDark ? 'bg-zinc-900 border-zinc-700' : 'bg-white border-zinc-200'
  }`

  return (
    <div className={card} role="dialog" aria-labelledby="gw-settings-title">
      <div className="flex items-center gap-2 mb-3">
        <Server className="size-4 text-blue-500" />
        <h2 id="gw-settings-title" className="text-sm font-semibold">
          MLXGateway
        </h2>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onClose}
          className={`text-xs px-2 py-1 rounded ${isDark ? 'hover:bg-zinc-800' : 'hover:bg-zinc-100'}`}
        >
          Close
        </button>
      </div>
      <p className={`text-[10px] mb-3 ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
        Start the gateway (e.g. ./start.sh). CORS is enabled — the editor calls the API from the browser.
      </p>
      <label className="block space-y-1 mb-2">
        <span className={`text-[10px] uppercase ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>Base URL</span>
        <input
          value={localUrl}
          onChange={(e) => setLocalUrl(e.target.value)}
          onBlur={() => setBaseUrl(localUrl)}
          className={`w-full px-2 py-1.5 rounded border text-xs font-mono ${
            isDark ? 'bg-zinc-800 border-zinc-600 text-zinc-100' : 'bg-white border-zinc-300'
          }`}
        />
      </label>
      <label className="block space-y-1 mb-3">
        <span className={`text-[10px] uppercase ${isDark ? 'text-zinc-500' : 'text-zinc-500'}`}>
          API Key (optional)
        </span>
        <input
          type="password"
          value={localKey}
          onChange={(e) => setLocalKey(e.target.value)}
          onBlur={() => setApiKey(localKey)}
          placeholder="Bearer token if server requires it"
          className={`w-full px-2 py-1.5 rounded border text-xs ${
            isDark ? 'bg-zinc-800 border-zinc-600 text-zinc-100' : 'bg-white border-zinc-300'
          }`}
        />
      </label>
      <div className="flex flex-wrap gap-2 items-center mb-3">
        <button
          type="button"
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            try {
              setBaseUrl(localUrl)
              setApiKey(localKey)
              await checkHealth()
              await fetchModels()
            } finally {
              setBusy(false)
            }
          }}
          className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs ${
            isDark ? 'bg-blue-600 text-white' : 'bg-blue-600 text-white'
          } disabled:opacity-50`}
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
          Test connection
        </button>
        <span
          className={`text-[11px] font-medium ${
            connected ? 'text-green-500' : isDark ? 'text-zinc-500' : 'text-zinc-500'
          }`}
        >
          {connected ? 'Connected' : 'Not connected'}
        </span>
      </div>
      {lastHealthError && (
        <div className="text-[10px] text-red-400 mb-2 break-all">{lastHealthError}</div>
      )}
      <div className={`text-[10px] font-medium mb-1 ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
        Models ({models.length})
      </div>
      <div
        className={`max-h-36 overflow-y-auto rounded border text-[10px] font-mono p-2 ${
          isDark ? 'border-zinc-700 bg-zinc-950' : 'border-zinc-200 bg-zinc-50'
        }`}
      >
        {models.length === 0 ? (
          <span className={isDark ? 'text-zinc-500' : 'text-zinc-500'}>Run “Test connection” to load.</span>
        ) : (
          models.slice(0, 80).map((m) => <div key={m.id}>{m.id}</div>)
        )}
        {models.length > 80 && <div className="opacity-60">…</div>}
      </div>
    </div>
  )
}
