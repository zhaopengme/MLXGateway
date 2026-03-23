# MLXGateway

MLXGateway 是一个专为 **Apple Silicon (M系列芯片)** 设计的、兼容 **OpenAI API** 格式的本地大模型服务网关。

它将多个基于 MLX 框架的 AI 能力库整合在同一个 FastAPI 服务器下，让你能够通过一套标准的 API 接口，轻松调用本地运行的各类大模型（涵盖文本、视觉、音频、图像、视频）。

## 核心特性

- **OpenAI API 兼容**：完全兼容 OpenAI 的接入格式（`/v1/chat/completions` 等），可直接使用任何支持自定义 API 端点的客户端。
- **统一网关，五大模态**：
  - **LLM (文本大模型)**：基于 `mlx-lm`，支持流式输出和工具调用。
  - **VLM (视觉语言大模型)**：基于 `mlx-vlm`，支持图片、音频、视频输入。
  - **Audio (音频处理)**：基于 `mlx-audio`，提供 STT 语音识别和 TTS 语音合成（支持声音克隆）。
  - **Images (图像生成)**：基于 `mflux`，提供文生图和图片编辑。
  - **Video (视频生成)**：基于 `mlx-video`，支持 T2V（文生视频）、I2V（图生视频，支持首帧+尾帧双控）、A2V（音频驱动视频）。
  - **Embeddings (文本向量化)**：基于 `mlx-embeddings`，提供文本转换向量服务。
- **高效的模型缓存与调度**：LRU 缓存机制管理模型权重，支持 Prompt Cache 自动落盘与恢复。
- **安全的 GPU 并发控制**：按推理类型（LLM/Embedding/Image/Audio/Video）独立的信号量调度，避免跨类型请求互相阻塞。

## 安装

> 要求：macOS + Apple Silicon (M1/M2/M3/M4)

```bash
git clone https://github.com/zhaopengme/MLXGateway.git
cd MLXGateway

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

## 启动

```bash
mlxgateway --host 0.0.0.0 --port 8008 --log-level info
```

或使用 `start.sh`（推荐，从 `.env` 读取 API Key）：

```bash
./start.sh
```

## 下载模型

使用 HuggingFace CLI 预下载模型：

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

## API 示例

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

**声音克隆**：将参考音频放到 `ref/` 目录（如 `ref/myvoice.ogg`），`voice` 填文件名（不含扩展名）：

```bash
curl -X POST http://localhost:8008/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token" \
  -d '{
    "model": "mlx-community/fish-audio-s2-pro-bf16",
    "input": "这是一段声音克隆测试。",
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

**文生视频 (T2V)**：

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

**图生视频 - 首帧 (I2V)**：

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

**图生视频 - 首帧+尾帧 (I2V dual-frame)**：

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

**音频驱动视频 (A2V)**：

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

**A2V + I2V 组合（音频驱动 + 首尾帧控制）**：

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

#### Video API 参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `prompt` | (必填) | 视频描述（建议用英文） |
| `model` | `prince-canuma/LTX-2.3-distilled` | 视频模型 |
| `width` / `height` | `512` | 分辨率（必须被 32 整除，上限 2048） |
| `num_frames` | `97` | 帧数（必须为 1+8k，如 9/17/25/.../257） |
| `fps` | `24` | 帧率 |
| `pipeline` | `distilled` | 生成管线：`distilled`(快) / `dev`(高质量) / `dev-two-stage` / `dev-two-stage-hq` |
| `cfg_scale` | `3.0` | CFG 引导强度（dev pipeline 生效） |
| `seed` | 随机 | 随机种子 |
| `image` / `image_url` | - | 首帧图片（base64 或 URL） |
| `end_image` / `end_image_url` | - | 尾帧图片（base64 或 URL） |
| `image_strength` | `1.0` | 图片条件强度 |
| `audio` | `true` | 是否生成同步音频（A2V 模式自动关闭） |
| `audio_file` / `audio_file_url` | - | A2V 输入音频（base64 或 URL） |
| `audio_start_time` | `0.0` | A2V 音频起始偏移（秒） |
| `response_format` | `url` | 返回格式：`url` 或 `b64_json` |
| `tiling` | `auto` | VAE tiling 模式 |

## 配置参数

| CLI 参数 | 环境变量 | 说明 | 默认值 |
|:---|:---|:---|:---|
| `--host` | `HOST` | 监听地址 | `127.0.0.1` |
| `--port` | `PORT` | 监听端口 | `8008` |
| `--api-key` | `API_KEY` | API 密钥（可选） | 禁用 |
| `--max-models` | `MAX_MODELS` | 最大缓存模型数 | `4` |
| `--model-cache-ttl` | `MODEL_CACHE_TTL` | 模型空闲过期时间（秒） | `600` |
| `--max-concurrent` | `MAX_CONCURRENT` | GPU 最大并行推理数 | `1` |
| `--request-timeout` | `REQUEST_TIMEOUT` | GPU 队列等待超时（秒） | `300` |
| `--ref-audio` | `REF_AUDIO_PATH` | TTS 参考音频目录 | `ref/` |

## API 端点一览

| 端点 | 说明 |
|:---|:---|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 列出可用模型 |
| `POST /v1/chat/completions` | 对话补全（文本/多模态/流式） |
| `POST /v1/embeddings` | 文本嵌入向量 |
| `POST /v1/audio/transcriptions` | 语音转文字 (STT) |
| `POST /v1/audio/speech` | 文字转语音 (TTS) |
| `POST /v1/images/generations` | 文生图 |
| `POST /v1/images/edits` | 图片编辑 |
| `POST /v1/videos/generations` | 视频生成（T2V / I2V / A2V） |
| `GET /static/...` | 静态文件（生成的图片/视频，无需认证） |

启动后访问 [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs) 查看交互式 API 文档。

## 贡献

欢迎提交 Issue 和 Pull Request！
