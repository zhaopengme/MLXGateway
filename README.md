# MLXGateway

MLXGateway is an **OpenAI API-compatible** local AI gateway designed for **Apple Silicon (M-series chips)**. It integrates multiple MLX-based AI libraries into a single FastAPI server, providing a unified API for text, vision, audio, image, and video models running locally.

## Features

- **OpenAI API Compatible**: Works with any client that supports custom API endpoints (NextChat, Cherry Studio, OpenAI SDK, etc.)
- **Six Modalities in One Gateway**:
  - **LLM**: Powered by `mlx-lm` with streaming, tool calling, and thinking/reasoning support
  - **VLM**: Powered by `mlx-vlm` with image, audio, and video input
  - **Audio**: Powered by `mlx-audio` for STT and TTS with voice cloning
  - **Images**: Powered by `mflux` for text-to-image and image editing (FLUX.1/FLUX.2/Kontext/Qwen)
  - **Video**: Powered by `mlx-video` for T2V, I2V (first + last frame), and A2V (audio-driven)
  - **Embeddings**: Powered by `mlx-embeddings` for text vectorization with dynamic batching
- **Two Run Modes**:
  - **Single-process** (`--mode single`): All modalities in one process, simple setup
  - **Multi-process** (`--mode multi`): 6 isolated worker processes behind a proxy, video generation never blocks embeddings
- **Model Caching**: LRU cache with TTL eviction and automatic prompt cache persistence
- **GPU Concurrency Control**: Per-type semaphores (LLM/Embedding/Image/Audio/Video) prevent cross-type request starvation
- **Static File Cleanup**: Periodic cleanup of generated images/videos (1-hour TTL)

## Installation

> Requires: macOS with Apple Silicon (M1/M2/M3/M4), Python >= 3.11

```bash
git clone https://github.com/zhaopengme/MLXGateway.git
cd MLXGateway

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Quick Start

### Single-process mode

```bash
mlxgateway --host 0.0.0.0 --port 8008 --log-level info
```

### Multi-process mode (recommended for production)

```bash
mlxgateway --host 0.0.0.0 --port 8008 --mode multi
```

Or use `start.sh` (recommended, reads API key from `.env`, defaults to multi-process mode):

```bash
./start.sh
```

Multi-process mode spawns 6 isolated workers (chat, embedding, stt, tts, image, video) behind a reverse proxy. Each worker has its own Metal GPU context, so long-running tasks (e.g., video generation) never block fast requests (e.g., embeddings).

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

### Downloading GGUF Models

```bash
# Download a specific quantization (recommended: Q4_K_M)
huggingface-cli download unsloth/Qwen3-8B-GGUF \
  --include "qwen3-8b-q4_k_m.gguf" --local-dir-use-symlinks False

# Or let MLXGateway download automatically at first use
# (just reference the repo in the API call)
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

### Chat with a GGUF Model

GGUF models are identified automatically by repo name or file suffix. No extra setup is needed — the first request downloads and caches the model.

**Auto-select quantization from HF repo (picks Q4_K_M by default):**

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "unsloth/Qwen3-8B-GGUF",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Pin a specific quantization level:**

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "unsloth/Qwen3-8B-GGUF:Q5_K_M",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Use a local GGUF file:**

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "/Users/me/models/qwen3-8b-q4_k_m.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Supported GGUF model_id formats:**

| Format | Example |
|:---|:---|
| Local file | `/path/to/model.gguf` |
| HF repo (auto-pick quant) | `unsloth/Qwen3-8B-GGUF` |
| HF repo + quant tag | `unsloth/Qwen3-8B-GGUF:Q4_K_M` |
| HF repo + exact filename | `unsloth/Qwen3-8B-GGUF:qwen3-8b-q4_k_m.gguf` |

**Quantization preference order** (when auto-selecting): Q4_K_M → Q5_K_M → Q4_K_S → Q5_K_S → Q4_0 → Q8_0 → Q6_K → Q3_K_M → Q2_K

**Supported GGUF architectures:** llama, qwen2, qwen3, phi, phi3, gemma, gemma2, mistral

### Text-to-Speech

```bash
curl -X POST http://localhost:8008/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/fish-audio-s2-pro-bf16",
    "input": "Hello, this is a speech synthesis test.",
    "voice": "liuyifei",
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

### Image Editing

Upload one or more source images and a prompt to create edited images:

```bash
curl -X POST http://localhost:8008/v1/images/edits \
  -H "Authorization: Bearer token" \
  -F "prompt=Make it look like a watercolor painting" \
  -F "model=flux2-klein-9b-edit" \
  -F "image[]=@source.png" \
  -F "response_format=b64_json"
```

### Video Generation

Video generation supports two endpoints: a JSON endpoint for base64/URL input, and an upload endpoint for direct file uploads.

#### JSON Endpoint (`POST /v1/videos/generations`)

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
    "prompt": "The girl speaks while looking at the ocean, mouth moving naturally, wind blowing hair",
    "image_url": "http://localhost:8008/static/first.jpeg",
    "end_image_url": "http://localhost:8008/static/end.png",
    "audio_file_url": "http://localhost:8008/static/speech.wav",
    "num_frames": 97,
    "width": 512,
    "height": 320
  }'
```

#### File Upload Endpoint (`POST /v1/videos/generations/upload`)

Upload image and audio files directly via multipart/form-data instead of base64/URL:

```bash
curl -X POST http://localhost:8008/v1/videos/generations/upload \
  -H "Authorization: Bearer token" \
  -F "prompt=A fox walking gracefully through a snowy forest" \
  -F "image=@first_frame.jpg" \
  -F "end_image=@last_frame.jpg" \
  -F "width=448" \
  -F "height=768" \
  -F "num_frames=121"
```

Upload with audio file (A2V + I2V):

```bash
curl -X POST http://localhost:8008/v1/videos/generations/upload \
  -H "Authorization: Bearer token" \
  -F "prompt=A narrator explaining wildlife" \
  -F "image=@first_frame.jpg" \
  -F "audio_file=@narration.wav" \
  -F "num_frames=97"
```

#### Video API Parameters

| Parameter | Default | Description |
|:---|:---|:---|
| `prompt` | (required) | Video description (English recommended) |
| `model` | `prince-canuma/LTX-2.3-distilled` | Video model |
| `text_encoder_repo` | `mlx-community/gemma-3-12b-it-bf16` | Text encoder model (set null for bundled encoders) |
| `width` / `height` | `512` | Resolution (must be divisible by 32, max 2048) |
| `num_frames` | `97` | Frame count (must be 1+8k, e.g., 9/17/25/.../257) |
| `fps` | `24` | Frames per second |
| `pipeline` | `distilled` | Pipeline: `distilled` (fast) / `dev` (quality) / `dev-two-stage` / `dev-two-stage-hq` |
| `cfg_scale` | `3.0` | CFG guidance scale (effective with dev pipeline) |
| `seed` | random | Random seed for reproducibility |
| `image` / `image_url` | - | First frame image (base64/URL for JSON, file for upload) |
| `end_image` / `end_image_url` | - | Last frame image (base64/URL for JSON, file for upload) |
| `image_strength` | `1.0` | Image conditioning strength |
| `audio` | `true` | Generate synchronized audio (auto-disabled in A2V mode) |
| `audio_cfg_scale` | `7.0` | CFG guidance scale for audio generation |
| `audio_file` / `audio_file_url` | - | A2V input audio (base64/URL for JSON, file for upload) |
| `audio_start_time` | `0.0` | A2V audio start offset (seconds) |
| `response_format` | `url` | Response format: `url` or `b64_json` |
| `tiling` | `auto` | VAE tiling mode: `auto` / `none` / `conservative` / `aggressive` / `spatial` / `temporal` |

## Configuration

| CLI Argument | Env Variable | Description | Default |
|:---|:---|:---|:---|
| `--host` | `HOST` | Bind address | `127.0.0.1` |
| `--port` | `PORT` | Bind port | `8008` |
| `--mode` | - | Run mode: `single` or `multi` | `single` |
| `--api-key` | `API_KEY` | API key authentication (optional) | disabled |
| `--log-level` | `LOG_LEVEL` | Logging level: `debug`/`info`/`warning`/`error`/`critical` | `info` |
| `--max-models` | `MAX_MODELS` | Max cached models | `4` |
| `--model-cache-ttl` | `MODEL_CACHE_TTL` | Model idle expiry (seconds) | `600` |
| `--model-list-cache` | `MODEL_LIST_CACHE_TTL` | Model list cache TTL (seconds) | `600` |
| `--max-concurrent` | `MAX_CONCURRENT` | Max concurrent GPU inference per type | `1` |
| `--request-timeout` | `REQUEST_TIMEOUT` | GPU queue wait timeout (seconds) | `300` |
| `--ref-audio` | `REF_AUDIO_PATH` | TTS reference audio directory | `ref/` |
| `--routers` | - | Comma-separated routers to enable (e.g., `chat,embedding`) | `all` |
| - | `ENABLE_CACHE` | Enable prompt caching (`true`/`false`) | `true` |
| - | `DEFAULT_MAX_KV_SIZE` | Default KV cache size (optional) | auto |

## API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/models/{model_id}` | GET | Get model details |
| `/v1/chat/completions` | POST | Chat completions (text/multimodal/streaming/tool calling) |
| `/v1/embeddings` | POST | Text embeddings (with dynamic batching) |
| `/v1/audio/transcriptions` | POST | Speech-to-text (STT) |
| `/v1/audio/speech` | POST | Text-to-speech (TTS) with voice cloning |
| `/v1/images/generations` | POST | Image generation |
| `/v1/images/edits` | POST | Image editing (multipart/form-data) |
| `/v1/videos/generations` | POST | Video generation via JSON (T2V / I2V / A2V) |
| `/v1/videos/generations/upload` | POST | Video generation via file upload (multipart/form-data) |
| `/static/...` | GET | Static files (generated images/videos, no auth required) |

Visit [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs) after starting the server for interactive API documentation.

## Contributing

Issues and Pull Requests are welcome!
