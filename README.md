# MLXGateway

OpenAI-compatible API server for MLX framework. All-in-one package combining mlx-lm, mlx-vlm, mlx-audio, mlx-embeddings, and mflux.

## Features

- **Chat Completions** - Text generation with streaming support via mlx-lm
- **Vision** - Image understanding and analysis via mlx-vlm
- **Text-to-Speech** - Audio generation from text via mlx-audio
- **Speech-to-Text** - Transcription and translation via mlx-audio
- **Embeddings** - Text embeddings via mlx-embeddings
- **Image Generation** - Text-to-image using mflux
- **Model Management** - Model caching with configurable TTL and size limits
- **Tool and Prompt Cache Support** - Using mlx-lm's tool calling and prompt cache support.
- **OpenAI Compatible** - OpenAI API endpoints

## Installation
clone this project

```bash
python3 -m venv ~/mlxgateway
source ~/mlxgateway/bin/activate

pip install -e .
```

## Usage

download model with huggingface cli 
```bash
hf download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

```bash
mlxgateway
```

Server starts at `http://localhost:8008`

### Options

```bash
mlxgateway --host 0.0.0.0 --port 8008 --log-level debug --max-models 4 --model-cache-ttl 600
```

## API Examples

### Chat Completions

```bash
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/MiroThinker-1.7-mini-mlx-4Bit",
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
    "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "input": "Hello, this is a speech synthesis test.",
    "voice": "Serena",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Embeddings

```bash
curl http://localhost:8008/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/bge-m3-mlx-4bit",
    "input": "Hello world"
  }'
```

Batch input:

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
  }' \
  --output response.json
```

## Thanks to

- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Language models
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Audio models (TTS & STT)
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Text embeddings
- [mflux](https://github.com/filipstrand/mflux) - Image generation
