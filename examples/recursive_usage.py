#!/usr/bin/env python3
"""
Examples of recursive Claude-calling-Claude patterns using claude-wrapper.

This demonstrates common patterns where Claude analyzes its own output,
performs iterative refinement, or chains multiple analysis steps.
"""

import asyncio
from claude_wrapper.core import (
    ClientMode,
    UnifiedClaudeClient,
    create_claude_client,
    create_claude_wrapper_client,
    create_api_client,
    RecursionError,
)


async def example_basic_recursion():
    """Basic example of Claude calling itself for analysis."""
    print("=== Basic Recursion Example ===")

    # Create a client with recursion safeguards
    client = create_claude_client(max_recursion_depth=3)

    try:
        # Initial analysis
        code_to_analyze = """
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr
        """

        print("Step 1: Initial code analysis")
        analysis = await client.chat(f"Analyze this Python code for efficiency and suggest improvements:\n{code_to_analyze}")
        print("Analysis:", analysis[:200] + "..." if len(analysis) > 200 else analysis)

        print("\nStep 2: Meta-analysis of the analysis")
        meta_analysis = await client.chat(f"Review this code analysis for completeness and accuracy:\n{analysis}")
        print("Meta-analysis:", meta_analysis[:200] + "..." if len(meta_analysis) > 200 else meta_analysis)

        print("\nStep 3: Final synthesis")
        synthesis = await client.chat(f"Create a final summary combining these analyses:\nOriginal: {analysis}\nReview: {meta_analysis}")
        print("Synthesis:", synthesis[:200] + "..." if len(synthesis) > 200 else synthesis)

        # Check recursion info
        info = client.get_recursion_info()
        print(f"\nRecursion tracking: depth={info['current_depth']}, max={info['max_depth']}")

    except RecursionError as e:
        print(f"Recursion limit reached: {e}")


async def example_iterative_improvement():
    """Example of iterative improvement using recursion."""
    print("\n=== Iterative Improvement Example ===")

    client = create_claude_wrapper_client(max_recursion_depth=5)

    initial_text = "The weather is nice today. It's sunny outside."
    current_text = initial_text

    print(f"Original text: {initial_text}")

    for iteration in range(3):
        print(f"\nIteration {iteration + 1}:")
        try:
            improved_text = await client.chat(
                f"Improve this text by making it more descriptive and engaging (keep it concise):\n{current_text}"
            )
            print(f"Improved: {improved_text}")
            current_text = improved_text

        except RecursionError as e:
            print(f"Stopped due to recursion limit: {e}")
            break

    print(f"\nFinal result: {current_text}")


async def example_mixed_mode_recursion():
    """Example using different client modes in a recursive pattern."""
    print("\n=== Mixed Mode Recursion Example ===")

    # Create clients for different modes
    api_client = create_api_client(max_recursion_depth=4)
    cli_client = create_claude_client(max_recursion_depth=4)

    try:
        # Step 1: API client for initial analysis
        print("Step 1: API analysis")
        api_response = await api_client.chat("Explain the concept of recursion in programming in simple terms.")
        print("API Response:", api_response[:150] + "..." if len(api_response) > 150 else api_response)

        # Step 2: CLI client for verification
        print("\nStep 2: CLI verification")
        cli_response = await cli_client.chat(f"Rate the accuracy of this explanation (1-10) and suggest improvements:\n{api_response}")
        print("CLI Response:", cli_response[:150] + "..." if len(cli_response) > 150 else cli_response)

        # Step 3: Back to API for final synthesis
        print("\nStep 3: Final synthesis")
        final_response = await api_client.chat(f"Create an improved explanation based on this feedback:\nOriginal: {api_response}\nFeedback: {cli_response}")
        print("Final Response:", final_response[:150] + "..." if len(final_response) > 150 else final_response)

    except RecursionError as e:
        print(f"Recursion limit reached: {e}")


async def example_streaming_recursion():
    """Example of recursive calls with streaming responses."""
    print("\n=== Streaming Recursion Example ===")

    client = create_api_client(max_recursion_depth=2)

    try:
        print("Step 1: Streaming initial response")
        response_parts = []
        async for chunk in client.stream_chat("Write a very short story about a robot learning to paint (2-3 sentences)."):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)

        initial_story = "".join(response_parts)
        print(f"\n\nInitial story captured: {initial_story}")

        print("\nStep 2: Streaming improvement")
        async for chunk in client.stream_chat(f"Improve this story by adding more emotion:\n{initial_story}"):
            print(chunk, end="", flush=True)
        print("\n")

    except RecursionError as e:
        print(f"Recursion limit reached: {e}")


async def example_recursion_safeguards():
    """Demonstrate recursion safeguards and protection mechanisms."""
    print("\n=== Recursion Safeguards Example ===")

    # Create a client with a very low recursion limit for demonstration
    client = create_claude_client(max_recursion_depth=2)

    print("Testing recursion limits...")

    try:
        # First call - should work
        print("Call 1: Success")
        await client.chat("Say hello")

        # Second nested call - should work
        print("Call 2: Success")
        await client.chat("Analyze the greeting 'hello' for friendliness")

        # Third nested call - should hit the limit
        print("Call 3: Testing limit...")
        await client.chat("Provide a meta-analysis of greeting analysis")

    except RecursionError as e:
        print(f"✓ Recursion safeguard triggered: {e}")

    # Demonstrate recursion info
    info = client.get_recursion_info()
    print(f"Recursion info: {info}")


async def example_context_isolation():
    """Demonstrate that different execution contexts have isolated recursion tracking."""
    print("\n=== Context Isolation Example ===")

    client = create_claude_client(max_recursion_depth=3)

    async def task_a():
        """Task A with its own recursion context."""
        print("Task A: Starting")
        await client.chat("Task A: Analyze this concept: 'artificial intelligence'")
        print("Task A: Step 1 complete")
        await client.chat("Task A: Provide a deeper analysis")
        print("Task A: Step 2 complete")
        return "Task A completed"

    async def task_b():
        """Task B with its own recursion context."""
        print("Task B: Starting")
        await client.chat("Task B: Explain machine learning")
        print("Task B: Step 1 complete")
        await client.chat("Task B: Give examples")
        print("Task B: Step 2 complete")
        return "Task B completed"

    # Run tasks concurrently - they should have isolated recursion tracking
    try:
        results = await asyncio.gather(task_a(), task_b())
        print(f"Results: {results}")
    except RecursionError as e:
        print(f"Recursion error: {e}")


async def example_error_handling():
    """Demonstrate error handling in recursive scenarios."""
    print("\n=== Error Handling Example ===")

    client = create_claude_client(max_recursion_depth=3, retry_attempts=2)

    # Test with recursion disabled
    client_no_recursion = UnifiedClaudeClient(
        mode=ClientMode.CLAUDE_CLI,
        enable_recursion_tracking=False
    )

    print("Client with recursion tracking disabled:")
    info = client_no_recursion.get_recursion_info()
    print(f"Tracking enabled: {info['tracking_enabled']}")

    print("\nClient with recursion tracking enabled:")
    info = client.get_recursion_info()
    print(f"Tracking enabled: {info['tracking_enabled']}")
    print(f"Max depth: {info['max_depth']}")
    print(f"Current depth: {info['current_depth']}")


async def main():
    """Run all examples."""
    print("Claude-Wrapper Recursive Usage Examples")
    print("=" * 50)

    examples = [
        example_basic_recursion,
        example_iterative_improvement,
        example_mixed_mode_recursion,
        example_streaming_recursion,
        example_recursion_safeguards,
        example_context_isolation,
        example_error_handling,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Example failed: {e}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    # Note: These examples assume you have Claude CLI installed and authenticated
    # or a claude-wrapper API server running
    print("Starting recursive usage examples...")
    print("Note: Examples may be simulated if Claude CLI is not available")

    asyncio.run(main())
