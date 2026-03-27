import { getGatewayBaseUrl, getGatewayHeaders } from '../stores/gatewayStore'

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

async function parseErrorMessage(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: { message?: string }; detail?: unknown }
    if (j.error?.message) return j.error.message
    if (typeof j.detail === 'string') return j.detail
    return `HTTP ${res.status}`
  } catch {
    return `HTTP ${res.status}`
  }
}

export async function generateImage(params: {
  model: string
  prompt: string
  size?: string
  n?: number
}): Promise<{ url: string }[]> {
  const base = getGatewayBaseUrl()
  const res = await fetch(`${base}/v1/images/generations`, {
    method: 'POST',
    headers: getGatewayHeaders(true) as HeadersInit,
    body: JSON.stringify({
      model: params.model,
      prompt: params.prompt,
      size: params.size ?? '1024x1024',
      n: params.n ?? 1,
      response_format: 'url',
    }),
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const j = (await res.json()) as { data?: Array<{ url?: string }> }
  const data = j.data ?? []
  return data.map((d) => ({ url: d.url ?? '' })).filter((d) => d.url)
}

export async function editImage(params: {
  model: string
  prompt: string
  images: File[]
  response_format?: string
}): Promise<{ url: string }[]> {
  const base = getGatewayBaseUrl()
  const fd = new FormData()
  fd.set('prompt', params.prompt)
  fd.set('model', params.model)
  fd.set('response_format', params.response_format ?? 'url')
  for (const f of params.images) {
    fd.append('image[]', f)
  }
  const headers = getGatewayHeaders(false) as Record<string, string>
  delete (headers as Record<string, string>)['Content-Type']
  const h2: HeadersInit = {}
  if (headers.Authorization) h2.Authorization = headers.Authorization
  const res = await fetch(`${base}/v1/images/edits`, {
    method: 'POST',
    headers: h2,
    body: fd,
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const j = (await res.json()) as { data?: Array<{ url?: string }> }
  const data = j.data ?? []
  return data.map((d) => ({ url: d.url ?? '' })).filter((d) => d.url)
}

export async function generateVideo(
  body: Record<string, unknown>,
): Promise<{ url: string }> {
  const base = getGatewayBaseUrl()
  const res = await fetch(`${base}/v1/videos/generations`, {
    method: 'POST',
    headers: getGatewayHeaders(true) as HeadersInit,
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const j = (await res.json()) as { data?: Array<{ url?: string }> }
  const url = j.data?.[0]?.url
  if (!url) throw new Error('No video URL in response')
  return { url }
}

export async function chatCompletions(params: {
  model: string
  messages: ChatMessage[]
  temperature?: number
  stream?: false
}): Promise<string> {
  const base = getGatewayBaseUrl()
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: getGatewayHeaders(true) as HeadersInit,
    body: JSON.stringify({
      model: params.model,
      messages: params.messages,
      temperature: params.temperature ?? 0.7,
      stream: false,
    }),
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const j = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>
  }
  const text = j.choices?.[0]?.message?.content
  if (typeof text !== 'string') throw new Error('No completion text')
  return text
}

export async function chatCompletionsStream(
  params: {
    model: string
    messages: ChatMessage[]
    temperature?: number
  },
  onDelta: (chunk: string) => void,
): Promise<void> {
  const base = getGatewayBaseUrl()
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: getGatewayHeaders(true) as HeadersInit,
    body: JSON.stringify({
      model: params.model,
      messages: params.messages,
      temperature: params.temperature ?? 0.7,
      stream: true,
    }),
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const parts = buf.split('\n')
    buf = parts.pop() ?? ''
    for (const line of parts) {
      const t = line.trim()
      if (!t.startsWith('data:')) continue
      const data = t.slice(5).trim()
      if (data === '[DONE]') continue
      try {
        const j = JSON.parse(data) as {
          choices?: Array<{ delta?: { content?: string | null } }>
        }
        const c = j.choices?.[0]?.delta?.content
        if (typeof c === 'string' && c) onDelta(c)
      } catch {
        /* skip malformed chunk */
      }
    }
  }
}

export async function textToSpeech(params: {
  model: string
  input: string
  voice: string
  response_format: string
}): Promise<Blob> {
  const base = getGatewayBaseUrl()
  const res = await fetch(`${base}/v1/audio/speech`, {
    method: 'POST',
    headers: getGatewayHeaders(true) as HeadersInit,
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  return res.blob()
}

export async function speechToText(file: File, model?: string): Promise<string> {
  const base = getGatewayBaseUrl()
  const fd = new FormData()
  fd.set('file', file)
  if (model) fd.set('model', model)
  const headers = getGatewayHeaders(false) as Record<string, string>
  const h2: HeadersInit = {}
  if (headers.Authorization) h2.Authorization = headers.Authorization
  const res = await fetch(`${base}/v1/audio/transcriptions`, {
    method: 'POST',
    headers: h2,
    body: fd,
  })
  if (!res.ok) throw new Error(await parseErrorMessage(res))
  const j = (await res.json()) as { text?: string }
  if (typeof j.text !== 'string') throw new Error('No transcription text')
  return j.text
}
