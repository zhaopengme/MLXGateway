# MLXGateway

MLXGateway 是一个专为 **Apple Silicon (M系列芯片)** 设计的、兼容 **OpenAI API** 格式的本地大模型服务网关。

它将多个基于 MLX 框架的 AI 能力库整合在同一个 FastAPI 服务器下，让你能够通过一套标准的 API 接口，轻松调用本地运行的各类大模型（涵盖文本、视觉、音频、图像）。

## 🌟 核心特性

- **OpenAI API 兼容**：完全兼容 OpenAI 的接入格式（`/v1/chat/completions` 等），可直接使用任何支持自定义 API 端点的客户端（如 NextChat, Cherry Studio, OpenAI 官方 SDK 等）。
- **统一大一统网关**：在一个服务中无缝集成四大模态：
  - **LLM (文本大模型)**：基于 `mlx-lm`，支持流式输出 (Streaming) 和工具调用 (Tool Calling)。
  - **VLM (视觉语言大模型)**：基于 `mlx-vlm`，支持通过提示词传入图片、音频或视频。
  - **Audio (音频/语音处理)**：基于 `mlx-audio`，提供 STT 语音识别 (`/v1/audio/transcriptions`) 和带有声音克隆能力的 TTS 语音合成 (`/v1/audio/speech`)。
  - **Images (图像生成与编辑)**：基于 `mflux`，提供生图 (`/v1/images/generations`) 和修图 (`/v1/images/edits`) 功能，并支持通过静态 URL 直接预览结果。
  - **Embeddings (文本向量化)**：基于 `mlx-embeddings`，提供文本转换向量服务 (`/v1/embeddings`)。
- **高效的模型缓存与调度**：
  - 采用模块化的 `OrderedDict` (LRU) 缓存机制管理所有加载的权重文件，避免重复读取硬盘引发的延迟。
  - 支持 Prompt Cache 自动落盘与恢复。
- **安全的 GPU 并发控制**：通过请求级别的显式 GPU Semaphore（信号量）调度，有效杜绝由于并发推理引发的内存溢出 (OOM) 崩溃问题。

## 🚀 安装

> 要求：macOS 设备且配备 Apple Silicon (M1/M2/M3/M4) 芯片。

建议使用虚拟环境：

```bash
# 克隆仓库
git clone https://github.com/zhaopengme/MLXGateway.git
cd MLXGateway

# 创建虚拟环境并激活
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .
```

## 💻 启动与配置

可以直接通过命令行启动服务：

```bash
mlxgateway --host 0.0.0.0 --port 8008 --log-level info
```

### 可用参数与环境变量配置

你可以通过命令行参数或同名的环境变量来配置网关行为：

| CLI 参数 | 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `--host` | `HOST` | 监听的主机地址 | `127.0.0.1` |
| `--port` | `PORT` | 监听的端口 | `8008` |
| `--api-key` | `API_KEY` | 设置 API 密钥身份验证（可选） | 禁用 |
| `--max-models` | `MAX_MODELS` | LLM/VLM 在内存中缓存的最大模型数量 | `4` |
| `--model-cache-ttl` | `MODEL_CACHE_TTL` | 模型的空闲过期释放时间 (秒) | `600` |
| `--max-concurrent` | `MAX_CONCURRENT` | 允许的 GPU 最大并行推理请求数 | `1` |
| `--request-timeout` | `REQUEST_TIMEOUT` | 请求进入 GPU 对列的等待超时 (秒) | `300` |
| `--ref-audio` | `REF_AUDIO_PATH` | TTS 声音克隆的参考音频目录或文件。请求指定的 `voice` 会在此目录下查找 `{voice}.wav` 或 `{voice}.ogg` 文件。 | `ref/` |

## 🔌 API 接口文档

启动服务后，你可以直接访问 [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs) 查看基于 Swagger UI 自动生成的交互式 API 文档。

主要的端点包括：
- `GET /v1/models` - 列出本地 Hugging Face 缓存中可用的模型。
- `POST /v1/chat/completions` - 对话补全（文本/多模态/流式）。
- `POST /v1/embeddings` - 获取文本嵌入向量。
- `POST /v1/audio/transcriptions` - 语音转文字。
- `POST /v1/audio/speech` - 文字转语音。
- `POST /v1/images/generations` - 文生图。
- `POST /v1/images/edits` - 图生图/图片编辑。

## 📝 贡献

欢迎提交 Issue 和 Pull Request 改进代码！
