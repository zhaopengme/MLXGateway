# MLXGateway

MLXGateway is an **OpenAI API-compatible** local AI gateway designed for **Apple Silicon (M-series chips)**. It integrates multiple MLX-based AI libraries into a single FastAPI server, providing a unified API for text, vision, audio, image, and video models running locally.

## Features

- **OpenAI API Compatible**: Works with any client that supports custom API endpoints (NextChat, Cherry Studio, OpenAI SDK, etc.)
- **Five Modalities in One Gateway**:
  - **LLM**: Powered by `mlx-lm` with streaming and tool calling support
  - **VLM**: Powered by `mlx-vlm` with image, audio, and video input
  - **Audio**: Powered by `mlx-audio` for STT and TTS with voice cloning
  - **Images**: Powered by `mflux` for text-to-image and image editing
  - **Video**: Powered by `mlx-video` for T2V, I2V (first + last frame), and A2V (audio-driven)
  - **Embeddings**: Powered by `mlx-embeddings` for text vectorization
- **Model Caching**: LRU cache with TTL eviction and automatic prompt cache persistence
- **GPU Concurrency Control**: Per-type semaphores (LLM/Embedding/Image/Audio/Video) prevent cross-type request starvation

## Installation

> Requires: macOS with Apple Silicon (M1/M2/M3/M4)

```bash
git clone https://github.com/zhaopengme/MLXGateway.git
cd MLXGateway

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Quick Start

```bash
mlxgateway --host 0.0.0.0 --port 8008 --log-level info
```

Or use `start.sh` (recommended, reads API key from `.env`):

```bash
./start.sh
```

## Download Models

Pre-download models with HuggingFace CLI:

```bash
# LLM
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit

# TTS
hf download mlx-community/fish-audio-s2-pro-bf16

# Embeddings
hf download mlx-community/bge-m3-mlx-4bit

# Video (LTX-2.3)
hf download prince-canuma/LTX-2.3-distilled
```

## API Examples

### Chat Completions

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Text-to-Speech

```bash
curl -X POST http://localhost:8008/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/fish-audio-s2-pro-bf16",
    "input": "Hello, this is a speech synthesis test.",
    "voice": "af_sky",
    "response_format": "wav"
  }' \
  --output speech.wav
```

**Voice Cloning**: Place reference audio in the `ref/` directory (e.g., `ref/myvoice.ogg`), then set `voice` to the filename without extension:

```bash
curl -X POST http://localhost:8008/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/fish-audio-s2-pro-bf16",
    "input": "This is a voice cloning test.",
    "voice": "myvoice",
    "response_format": "wav"
  }' \
  --output clone.wav
```

### Embeddings

```bash
curl http://localhost:8008/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/bge-m3-mlx-4bit",
    "input": ["Hello world", "How are you"]
  }'
```

### Image Generation

```bash
curl -X POST http://localhost:8008/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "black-forest-labs/FLUX.2-klein-4B",
    "prompt": "a cat sitting on a desk, studio lighting",
    "size": "1024x1024"
  }'
```

### Video Generation

**Text-to-Video (T2V)**:

```bash
curl -X POST http://localhost:8008/v1/videos/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "prompt": "Ocean waves crashing against rocks at golden hour, cinematic, slow motion, 4K",
    "num_frames": 97,
    "width": 768,
    "height": 512
  }'
```

**Image-to-Video - First Frame (I2V)**:

```bash
curl -X POST http://localhost:8008/v1/videos/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "prompt": "The scene slowly comes to life with gentle movement, cinematic",
    "image_url": "http://localhost:8008/static/first.jpeg",
    "num_frames": 97
  }'
```

**Image-to-Video - First + Last Frame (I2V Dual-Frame)**:

```bash
curl -X POST http://localhost:8008/v1/videos/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "prompt": "Smooth transition between the two scenes, cinematic motion",
    "image_url": "http://localhost:8008/static/first.jpeg",
    "end_image_url": "http://localhost:8008/static/end.png",
    "num_frames": 97
  }'
```

**Audio-to-Video (A2V)**:

```bash
curl -X POST http://localhost:8008/v1/videos/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "prompt": "A band playing music on stage, dynamic camera, cinematic",
    "audio_file_url": "http://localhost:8008/static/music.wav",
    "num_frames": 97
  }'
```

**A2V + I2V Combined (Audio-Driven + Frame Control)**:

```bash
curl -X POST http://localhost:8008/v1/videos/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "prompt": "The girl speaks while looking at the ocean, mouth moving naturally, wind blowing hair, Pixar 3D animation",
    "image_url": "http://localhost:8008/static/first.jpeg",
    "end_image_url": "http://localhost:8008/static/end.png",
    "audio_file_url": "http://localhost:8008/static/speech.wav",
    "num_frames": 97,
    "width": 512,
    "height": 320
  }'
```

#### Video API Parameters

| Parameter | Default | Description |
|:---|:---|:---|
| `prompt` | (required) | Video description (English recommended) |
| `model` | `prince-canuma/LTX-2.3-distilled` | Video model |
| `width` / `height` | `512` | Resolution (must be divisible by 32, max 2048) |
| `num_frames` | `97` | Frame count (must be 1+8k, e.g., 9/17/25/.../257) |
| `fps` | `24` | Frames per second |
| `pipeline` | `distilled` | Pipeline: `distilled` (fast) / `dev` (quality) / `dev-two-stage` / `dev-two-stage-hq` |
| `cfg_scale` | `3.0` | CFG guidance scale (effective with dev pipeline) |
| `seed` | random | Random seed for reproducibility |
| `image` / `image_url` | - | First frame image (base64 or URL) |
| `end_image` / `end_image_url` | - | Last frame image (base64 or URL) |
| `image_strength` | `1.0` | Image conditioning strength |
| `audio` | `true` | Generate synchronized audio (auto-disabled in A2V mode) |
| `audio_file` / `audio_file_url` | - | A2V input audio (base64 or URL) |
| `audio_start_time` | `0.0` | A2V audio start offset (seconds) |
| `response_format` | `url` | Response format: `url` or `b64_json` |
| `tiling` | `auto` | VAE tiling mode |

## Configuration

| CLI Argument | Env Variable | Description | Default |
|:---|:---|:---|:---|
| `--host` | `HOST` | Bind address | `127.0.0.1` |
| `--port` | `PORT` | Bind port | `8008` |
| `--api-key` | `API_KEY` | API key authentication (optional) | disabled |
| `--max-models` | `MAX_MODELS` | Max cached models | `4` |
| `--model-cache-ttl` | `MODEL_CACHE_TTL` | Model idle expiry (seconds) | `600` |
| `--max-concurrent` | `MAX_CONCURRENT` | Max concurrent GPU inference | `1` |
| `--request-timeout` | `REQUEST_TIMEOUT` | GPU queue wait timeout (seconds) | `300` |
| `--ref-audio` | `REF_AUDIO_PATH` | TTS reference audio directory | `ref/` |

## API Endpoints

| Endpoint | Description |
|:---|:---|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (text/multimodal/streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | Speech-to-text (STT) |
| `POST /v1/audio/speech` | Text-to-speech (TTS) |
| `POST /v1/images/generations` | Image generation |
| `POST /v1/images/edits` | Image editing |
| `POST /v1/videos/generations` | Video generation (T2V / I2V / A2V) |
| `GET /static/...` | Static files (generated images/videos, no auth required) |

Visit [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs) after starting the server for interactive API documentation.

## Contributing

Issues and Pull Requests are welcome!
