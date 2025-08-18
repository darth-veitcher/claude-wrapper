# Claude Wrapper

A simple Python wrapper for Claude CLI that seamlessly integrates Claude's capabilities into your Python applications. This package provides streaming support and an OpenAI-compatible API interface.

## ✨ Key Features

- **🔧 Simple Integration**: Use Claude CLI through Python with minimal setup
- **🚀 Streaming Support**: Real-time response streaming with proper async handling
- **🔌 OpenAI-Compatible API**: Drop-in replacement for OpenAI API clients
- **🔒 No API Keys**: Leverages Claude CLI's authentication - no separate API keys needed
- **⚡ Robust Error Handling**: Automatic retries and comprehensive error management
- **📊 Efficient**: Subprocess optimization with configurable timeouts and retries
- **🔄 Recursive Calling**: Built-in support for Claude-calling-Claude with safeguards

## 🚀 5-Minute Quickstart

### Prerequisites

1. **Install Claude CLI** (if not already installed):
```bash
npm install -g @anthropic-ai/claude-cli
# OR
brew install claude

# Authenticate once
claude login
```

2. **Install Claude Wrapper**:
```bash
# Modern approach (recommended)
uvx claude-wrapper --help  # Run directly without installation
# OR install globally with uv
uv tool install claude-wrapper

# Traditional approach (if uv not available)
pip install claude-wrapper
```

### Quick Examples

#### Example 1: Basic Chat (30 seconds)
```python
import asyncio
from claude_wrapper import ClaudeClient

async def quick_chat():
    client = ClaudeClient()

    # Simple question
    response = await client.chat("What is Python?")
    print(response)

asyncio.run(quick_chat())
```

#### Example 2: Streaming Responses (1 minute)
```python
import asyncio
from claude_wrapper import ClaudeClient

async def stream_example():
    client = ClaudeClient()

    print("Claude: ", end="")
    async for chunk in client.stream_chat("Tell me a short joke"):
        print(chunk, end="", flush=True)
    print()  # New line after response

asyncio.run(stream_example())
```

#### Example 3: CLI Usage (30 seconds)
```bash
# Using uvx (no installation required)
uvx claude-wrapper chat "What is the capital of France?"
uvx claude-wrapper chat "Explain quantum computing" --stream

# If installed with uv or pip
claude-wrapper chat "What is the capital of France?"
claude-wrapper chat "Explain quantum computing" --stream
```

#### Example 4: API Server with OpenAI Client (2 minutes)
```bash
# Terminal 1: Start the server (using uvx)
uvx claude-wrapper server

# Or if installed
claude-wrapper server
```

```python
# Terminal 2: Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No API key required!
)

# Works exactly like OpenAI
response = client.chat.completions.create(
    model="sonnet",
    messages=[
        {"role": "user", "content": "Write a haiku about Python"}
    ]
)

print(response.choices[0].message.content)
```

#### Example 5: Advanced Usage with Error Handling (1 minute)
```python
import asyncio
from claude_wrapper import ClaudeClient
from claude_wrapper.core.exceptions import ClaudeTimeoutError, ClaudeAuthError

async def robust_chat():
    client = ClaudeClient(timeout=30, retry_attempts=3)

    try:
        # Simple request
        response = await client.chat("Explain quantum computing")
        print(f"Response: {response}")

    except ClaudeAuthError:
        print("Please run 'claude login' first")
    except ClaudeTimeoutError:
        print("Request timed out, try increasing timeout")

asyncio.run(robust_chat())
```

#### Example 6: Recursive Claude-Calling-Claude (2 minutes)
```python
import asyncio
from claude_wrapper.core import create_claude_client, RecursionError

async def recursive_analysis():
    # Create client with recursion safeguards
    client = create_claude_client(max_recursion_depth=5)

    try:
        # Initial analysis
        analysis = await client.chat("Analyze this concept: machine learning")
        print(f"Initial: {analysis[:100]}...")

        # Meta-analysis (recursive call)
        meta = await client.chat(f"Review this analysis for accuracy: {analysis}")
        print(f"Meta-analysis: {meta[:100]}...")

        # Final synthesis
        final = await client.chat(f"Synthesize these insights: {analysis} | {meta}")
        print(f"Final: {final[:100]}...")

        # Check recursion depth
        info = client.get_recursion_info()
        print(f"Recursion depth: {info['current_depth']}/{info['max_depth']}")

    except RecursionError as e:
        print(f"Recursion limit safely reached: {e}")

asyncio.run(recursive_analysis())
```

## 📦 Installation

### 🚀 Modern Approach (Recommended)

**Option 1: Run without installing (uvx)**
```bash
# No installation needed - run directly from PyPI
uvx claude-wrapper chat "Hello Claude!"
uvx claude-wrapper server --port 8080
uvx claude-wrapper --help
```

**Option 2: Install globally with uv**
```bash
# Install as a global tool
uv tool install claude-wrapper

# Use anywhere
claude-wrapper chat "Hello Claude!"
```

**Option 3: Install in project with uv**
```bash
# Add to your project
uv add claude-wrapper

# Use in your project
uv run claude-wrapper chat "Hello Claude!"
```

### 📦 Traditional Approach

**From PyPI**
```bash
pip install claude-wrapper
```

**From Source**
```bash
git clone https://github.com/darth-veitcher/claude-wrapper
cd claude-wrapper
pip install -e .
```

### 🔧 Development Setup

```bash
# Clone repository
git clone https://github.com/darth-veitcher/claude-wrapper
cd claude-wrapper

# Install in development mode with uv (recommended)
uv sync --dev

# Run tests
uv run pytest

# Traditional development setup
pip install -e ".[dev]"
pytest
```

## 🎯 Core Capabilities

### What Claude Wrapper Does

✅ **Wraps Claude CLI** - Provides Python interface to Claude CLI commands
✅ **Handles Streaming** - Supports real-time streaming with `--output-format stream-json`
✅ **Provides API Server** - OpenAI-compatible REST API
✅ **Error Recovery** - Automatic retries with exponential backoff
✅ **Token Estimation** - Estimates token usage (word count × 1.3)

### Current Limitations

The wrapper works with Claude CLI's supported options:

- ✅ Uses `--print` flag for non-interactive responses
- ✅ Supports `--output-format stream-json` for streaming
- ⚠️ Parameters like `max_tokens`, `temperature`, `system_prompt` are not passed to CLI (Claude CLI doesn't support them yet)
- ⚠️ Token counting is estimated, not exact

## 🏗️ Architecture

```
Your Application
       ↓
┌─────────────────────────────────┐
│      Claude Wrapper             │
│                                 │
│  ┌──────────────────────────┐  │
│  │     Python Interface     │  │
│  │  • ClaudeClient          │  │
│  │  • Error Handling        │  │
│  └────────────┬─────────────┘  │
│               ↓                 │
│  ┌──────────────────────────┐  │
│  │    Subprocess Manager    │  │
│  │  • Async execution       │  │
│  │  • Retry logic           │  │
│  │  • Timeout handling      │  │
│  └────────────┬─────────────┘  │
└───────────────┼─────────────────┘
                ↓
        Claude CLI (--print)
                ↓
            Claude AI
```

## 📐 API Reference

### ClaudeClient

```python
class ClaudeClient:
    def __init__(
        self,
        claude_path: str = "claude",  # Path to Claude CLI
        timeout: float = 120.0,  # Command timeout in seconds
        retry_attempts: int = 3,  # Number of retries
        retry_delay: float = 1.0  # Delay between retries
    )
```

#### Methods

**chat(message, **kwargs)**
- Send a message and get a response
- Returns: `str`

**stream_chat(message, **kwargs)**
- Stream a response in real-time
- Returns: `AsyncGenerator[str, None]`
- Falls back to regular chat if streaming unavailable

**complete(prompt, **kwargs)**
- Simple completion
- Returns: `str`

**count_tokens(text)**
- Estimate token count
- Returns: `Dict[str, int]` with `tokens` and `words`

## 🔧 CLI Commands

### Modern uv/uvx Usage
```bash
# Run directly without installation (uvx)
uvx claude-wrapper chat "Your message"
uvx claude-wrapper chat "Your message" --stream
uvx claude-wrapper server --host 0.0.0.0 --port 8000
uvx claude-wrapper version

# If installed as uv tool
uv tool run claude-wrapper chat "Your message"
```

### Traditional Usage
```bash
# After pip/uv install
claude-wrapper chat "Your message"
claude-wrapper chat "Your message" --stream
claude-wrapper server --host 0.0.0.0 --port 8000
claude-wrapper version
```

## ⚙️ Configuration

### Environment Variables
```bash
export CLAUDE_WRAPPER_CLAUDE_PATH="/usr/local/bin/claude"
export CLAUDE_WRAPPER_TIMEOUT=120
export CLAUDE_WRAPPER_RETRY_ATTEMPTS=3
export CLAUDE_WRAPPER_API_KEY="optional-api-key"  # Optional API key for server
```

### Configuration File
Create `~/.claude-wrapper/config.yaml`:
```yaml
claude:
  path: /usr/local/bin/claude
  timeout: 120
  retry_attempts: 3

api:
  key: optional-api-key  # Optional API key for server
```

## 🧪 Testing

### Using uv (Recommended)
```bash
# Install dev dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_client.py -v

# Run without coverage
uv run pytest --no-cov
```

### Traditional Approach
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_client.py -v
```

## 🐛 Troubleshooting

### Claude CLI Not Found
```bash
# Check installation
which claude

# Set custom path
export CLAUDE_WRAPPER_CLAUDE_PATH=/path/to/claude
```

### Authentication Issues
```bash
# Re-authenticate
claude login
```

### Timeout Errors
```python
# Increase timeout
client = ClaudeClient(timeout=300)
```

### Streaming Not Working
The wrapper will automatically fall back to non-streaming mode if Claude CLI doesn't support streaming for your version.

## 📊 Performance Notes

- **First Message**: ~1-2 seconds (includes CLI startup)
- **Follow-up Messages**: ~0.8-1.5 seconds
- **Memory Usage**: ~50MB base
- **Token Estimation**: Approximately 1.3 tokens per word

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing`)
7. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for Claude and Claude CLI
- The Python async/await ecosystem
- FastAPI and Typer for excellent frameworks

---

**Built to make Claude accessible and efficient for Python developers.**
