# Infinite Canvas Editor

React + Vite + TypeScript canvas and timeline UI wired to **[MLXGateway](../README.md)** (OpenAI-compatible API on Apple Silicon).

## Run the editor

```bash
cd infinite-canvas-editor
npm install
npm run dev
```

## Run MLXGateway

From the repo root (see main [README](../README.md)):

```bash
./start.sh
# or: mlxgateway --host 0.0.0.0 --port 8008
```

In the editor, open **Gateway** in the header, set **Base URL** (e.g. `http://localhost:8008`), optional **API Key**, then **Test connection** to load `/v1/models` and verify `/health`.

## Node types

| Node        | API |
|------------|-----|
| **Image gen** | `POST /v1/images/generations` or `/v1/images/edits` (with incoming image nodes + “Use edits”) |
| **Video gen** | `POST /v1/videos/generations` (I2V / dual frame / A2V via graph: connect images + optional TTS for `audio_file` base64) |
| **Chat**      | `POST /v1/chat/completions` (streaming) |
| **TTS**       | `POST /v1/audio/speech` |
| **STT**       | `POST /v1/audio/transcriptions` |

Connect **text** / **prompt** nodes **into** generators for prompt text. Connect **media** / **image gen** outputs **into** **video gen** for `image_url` / `end_image_url`. Connect **TTS** into **video gen** for A2V (audio is sent as base64 in JSON).

Use **Timeline** on image/video nodes to append results to the bottom timeline; **Library** lists recent generations.

## Build

```bash
npm run build
```
