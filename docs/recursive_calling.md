# Recursive Claude Calling in Claude-Wrapper

Claude-wrapper provides comprehensive support for recursive Claude-calling-Claude scenarios through the `UnifiedClaudeClient`. This enables patterns where Claude analyzes its own output, performs iterative refinement, or chains multiple analysis steps.

## Overview

The recursive calling system includes:

- **Multiple Client Modes**: Support for Claude CLI, claude-wrapper CLI, and HTTP API
- **Recursion Safeguards**: Depth limiting, loop detection, and timeout protection
- **Context Isolation**: Proper isolation between concurrent recursive operations
- **Comprehensive Error Handling**: Graceful handling of recursion limits and failures

## Quick Start

```python
import asyncio
from claude_wrapper.core import create_claude_client, RecursionError

async def main():
    # Create a client with recursion safeguards
    client = create_claude_client(max_recursion_depth=5)

    try:
        # Initial analysis
        response1 = await client.chat("Analyze this code for improvements")

        # Meta-analysis (recursive call)
        response2 = await client.chat(f"Review this analysis: {response1}")

        # Final synthesis
        response3 = await client.chat(f"Combine these insights: {response1} | {response2}")

    except RecursionError as e:
        print(f"Recursion limit reached: {e}")

asyncio.run(main())
```

## Client Modes

### 1. Claude CLI Mode

Uses the original Claude CLI for recursive calls:

```python
from claude_wrapper.core import create_claude_client, ClientMode

client = create_claude_client(
    max_recursion_depth=10,
    timeout=120.0,
    retry_attempts=3
)

# Or explicit mode
client = UnifiedClaudeClient(
    mode=ClientMode.CLAUDE_CLI,
    claude_path="claude",  # Path to Claude CLI
    max_recursion_depth=10
)
```

### 2. Claude-Wrapper CLI Mode

Uses claude-wrapper CLI for recursive calls:

```python
from claude_wrapper.core import create_claude_wrapper_client

client = create_claude_wrapper_client(
    max_recursion_depth=8,
    claude_wrapper_path="claude-wrapper"
)
```

### 3. HTTP API Mode

Uses claude-wrapper HTTP API for recursive calls:

```python
from claude_wrapper.core import create_api_client

client = create_api_client(
    api_base_url="http://localhost:8000",
    api_key="your-api-key",  # Optional
    max_recursion_depth=12
)
```

## Recursion Safety Features

### Depth Limiting

Prevents infinite recursion by limiting call depth:

```python
client = create_claude_client(max_recursion_depth=5)

# This will raise RecursionError after 5 nested calls
try:
    await deeply_nested_function(client)
except RecursionError as e:
    print(f"Safely caught recursion: {e}")
```

### Loop Detection

Detects immediate recursion loops:

```python
# This pattern is automatically detected and prevented
async def problematic_pattern(client):
    response = await client.chat("Analyze this")
    # Immediate recursive call with same operation - detected as loop
    return await client.chat("Analyze this")  # RecursionError raised
```

### Context Isolation

Different execution contexts have isolated recursion tracking:

```python
async def task_a(client):
    # This task's recursion is tracked independently
    return await client.chat("Task A analysis")

async def task_b(client):
    # This task's recursion is tracked independently
    return await client.chat("Task B analysis")

# Both tasks can run concurrently with their own recursion limits
results = await asyncio.gather(task_a(client), task_b(client))
```

## Common Patterns

### 1. Iterative Analysis

```python
async def iterative_analysis(client, text, iterations=3):
    """Iteratively improve analysis through multiple passes."""
    current_analysis = text

    for i in range(iterations):
        current_analysis = await client.chat(
            f"Improve this analysis (iteration {i+1}): {current_analysis}"
        )

        # Check recursion depth
        info = client.get_recursion_info()
        print(f"Depth: {info['current_depth']}/{info['max_depth']}")

    return current_analysis
```

### 2. Code Review Chain

```python
async def code_review_chain(client, code):
    """Multi-stage code review process."""

    # Stage 1: Initial review
    review = await client.chat(f"Review this code:\n{code}")

    # Stage 2: Security analysis
    security = await client.chat(f"Analyze security issues in: {review}")

    # Stage 3: Performance analysis
    performance = await client.chat(f"Analyze performance issues in: {review}")

    # Stage 4: Final synthesis
    final = await client.chat(f"Synthesize these analyses:\nReview: {review}\nSecurity: {security}\nPerformance: {performance}")

    return final
```

### 3. Streaming Recursive Analysis

```python
async def streaming_recursive_analysis(client, prompt):
    """Use streaming for better user experience in recursive calls."""

    print("Initial analysis:")
    response_parts = []
    async for chunk in client.stream_chat(prompt):
        print(chunk, end="", flush=True)
        response_parts.append(chunk)

    initial_response = "".join(response_parts)

    print("\n\nMeta-analysis:")
    async for chunk in client.stream_chat(f"Analyze this response: {initial_response}"):
        print(chunk, end="", flush=True)
    print()
```

### 4. Mixed Mode Recursion

```python
async def mixed_mode_analysis(query):
    """Use different client modes for different stages."""

    # Use API for initial fast analysis
    api_client = create_api_client(max_recursion_depth=3)
    initial = await api_client.chat(query)

    # Use CLI for detailed verification
    cli_client = create_claude_client(max_recursion_depth=3)
    verification = await cli_client.chat(f"Verify this analysis: {initial}")

    # Back to API for final synthesis
    final = await api_client.chat(f"Combine: {initial} | {verification}")

    return final
```

## Recursion Monitoring

### Getting Recursion Information

```python
client = create_claude_client()

# Get current recursion state
info = client.get_recursion_info()
print(f"Tracking enabled: {info['tracking_enabled']}")
print(f"Current depth: {info['current_depth']}")
print(f"Max depth: {info['max_depth']}")
print(f"Call stack: {info['call_stack']}")
```

### Disabling Recursion Tracking

```python
# For high-performance scenarios where recursion isn't a concern
client = UnifiedClaudeClient(
    mode=ClientMode.CLAUDE_CLI,
    enable_recursion_tracking=False
)

info = client.get_recursion_info()
print(f"Tracking enabled: {info['tracking_enabled']}")  # False
```

## Error Handling

### Recursion Errors

```python
from claude_wrapper.core import RecursionError

try:
    await deeply_nested_recursive_function(client)
except RecursionError as e:
    print(f"Recursion limit exceeded: {e}")
    # Log the call stack for debugging
    info = client.get_recursion_info()
    print(f"Call stack: {info['call_stack']}")
```

### Client-Specific Errors

```python
from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeTimeoutError,
    ClaudeNotFoundError
)

try:
    response = await client.chat("Test message")
except ClaudeAuthError:
    print("Claude CLI not authenticated")
except ClaudeNotFoundError:
    print("Claude CLI not found")
except ClaudeTimeoutError:
    print("Request timed out")
except ClaudeExecutionError as e:
    print(f"Execution failed: {e}")
```

## Configuration Options

### UnifiedClaudeClient Parameters

```python
client = UnifiedClaudeClient(
    mode=ClientMode.CLAUDE_CLI,           # Client mode
    claude_path="claude",                 # Path to Claude CLI
    claude_wrapper_path="claude-wrapper", # Path to claude-wrapper CLI
    api_base_url="http://localhost:8000", # API server URL
    api_key=None,                        # Optional API key
    timeout=120.0,                       # Request timeout (seconds)
    retry_attempts=3,                    # Number of retry attempts
    max_recursion_depth=10,              # Maximum recursion depth
    enable_recursion_tracking=True,      # Enable recursion safety
)
```

## Performance Considerations

### Recursion Overhead

Recursion tracking adds minimal overhead:
- Context variable operations: ~0.1ms per call
- Thread-local storage: ~0.05ms per subprocess call
- Loop detection: ~0.02ms per call

### Memory Usage

- Context variables: ~100 bytes per recursion level
- Call stack tracking: ~50 bytes per call
- Thread-local storage: ~200 bytes per thread

### Optimization Tips

1. **Use appropriate recursion depths**: Start with lower limits (3-5) and increase as needed
2. **Disable tracking for high-performance scenarios**: Set `enable_recursion_tracking=False`
3. **Use streaming for better UX**: Stream responses in recursive scenarios
4. **Monitor recursion depth**: Check depth in loops to avoid hitting limits

## Best Practices

### 1. Set Appropriate Limits

```python
# For simple analysis chains
client = create_claude_client(max_recursion_depth=3)

# For complex iterative processes
client = create_claude_client(max_recursion_depth=8)

# For experimental deep recursion
client = create_claude_client(max_recursion_depth=15)
```

### 2. Handle Errors Gracefully

```python
async def safe_recursive_operation(client, data, max_attempts=3):
    """Safely perform recursive operation with fallbacks."""

    for attempt in range(max_attempts):
        try:
            return await recursive_analysis(client, data)
        except RecursionError:
            if attempt < max_attempts - 1:
                # Reduce complexity and try again
                data = simplify_data(data)
                print(f"Recursion limit hit, retrying with simpler data (attempt {attempt + 2})")
            else:
                # Fall back to non-recursive approach
                print("Using non-recursive fallback")
                return await simple_analysis(client, data)
```

### 3. Monitor Performance

```python
import time

async def monitored_recursive_call(client, message):
    """Monitor recursive call performance."""
    start_time = time.time()

    try:
        result = await client.chat(message)

        # Check recursion info
        info = client.get_recursion_info()
        elapsed = time.time() - start_time

        print(f"Call completed in {elapsed:.2f}s at depth {info['current_depth']}")
        return result

    except RecursionError as e:
        elapsed = time.time() - start_time
        print(f"Recursion limit reached after {elapsed:.2f}s: {e}")
        raise
```

### 4. Use Async Context Managers

```python
async def safe_api_recursion():
    """Use context managers for proper cleanup."""
    async with create_api_client(max_recursion_depth=5) as client:
        # Client will be properly closed even if recursion fails
        try:
            result = await recursive_api_calls(client)
            return result
        except RecursionError:
            print("Recursion limit reached, but client cleaned up properly")
            raise
```

## Troubleshooting

### Common Issues

1. **"Maximum recursion depth exceeded"**
   - Increase `max_recursion_depth` if legitimate deep recursion is needed
   - Check for infinite loops in your logic
   - Use recursion monitoring to understand call patterns

2. **"Detected infinite recursion loop"**
   - Review your recursive logic for immediate loops
   - Ensure recursive calls have different parameters or context

3. **"Claude CLI not found"**
   - Verify Claude CLI is installed and in PATH
   - Set correct `claude_path` parameter

4. **"API authentication failed"**
   - Verify API server is running
   - Check API key if using authentication
   - Confirm `api_base_url` is correct

### Debugging Recursion Issues

```python
# Enable detailed recursion logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor recursion state
async def debug_recursive_call(client, message):
    print("Before call:")
    info = client.get_recursion_info()
    print(f"  Depth: {info['current_depth']}")
    print(f"  Stack: {info['call_stack']}")

    try:
        result = await client.chat(message)

        print("After call:")
        info = client.get_recursion_info()
        print(f"  Depth: {info['current_depth']}")
        print(f"  Stack: {info['call_stack']}")

        return result

    except RecursionError as e:
        print(f"Recursion error: {e}")
        info = client.get_recursion_info()
        print(f"Final stack: {info['call_stack']}")
        raise
```

## Advanced Usage

### Custom Recursion Logic

```python
class CustomRecursiveClient(UnifiedClaudeClient):
    """Custom client with application-specific recursion logic."""

    async def smart_recursive_chat(self, message, max_iterations=5):
        """Recursive chat with intelligent stopping conditions."""

        responses = []
        current_message = message

        for iteration in range(max_iterations):
            try:
                response = await self.chat(current_message)
                responses.append(response)

                # Custom stopping condition
                if "FINAL_ANSWER" in response or len(response) < 50:
                    print(f"Stopping after {iteration + 1} iterations")
                    break

                # Prepare next iteration
                current_message = f"Elaborate on: {response}"

            except RecursionError:
                print(f"Recursion limit reached at iteration {iteration + 1}")
                break

        return responses
```

### Integration with Existing Code

```python
# Gradual migration from simple client
def migrate_to_recursive_client():
    """Example of migrating from ClaudeClient to UnifiedClaudeClient."""

    # Old approach
    # client = ClaudeClient()

    # New approach with recursion support
    client = create_claude_client(max_recursion_depth=5)

    # Same interface, but with recursion safety
    return client

# Wrapper for backward compatibility
async def compatible_chat(message, enable_recursion=False):
    """Backward compatible chat function."""
    if enable_recursion:
        client = create_claude_client()
    else:
        from claude_wrapper.core import ClaudeClient
        client = ClaudeClient()

    return await client.chat(message)
```

## Examples

See `examples/recursive_usage.py` for comprehensive examples of:
- Basic recursion patterns
- Iterative improvement
- Mixed-mode recursion
- Streaming recursive responses
- Error handling
- Context isolation
- Performance monitoring

## API Reference

For detailed API documentation, see the docstrings in:
- `claude_wrapper.core.unified_client.UnifiedClaudeClient`
- `claude_wrapper.core.http_client.ClaudeHTTPClient`
- `claude_wrapper.core.exceptions`
