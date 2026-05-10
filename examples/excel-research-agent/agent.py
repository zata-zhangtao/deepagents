"""Excel Research Agent.

Searches the web and generates Excel reports from the findings.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from tools import generate_excel, web_search

# Load environment variables from .env file if present
load_dotenv()


def create_excel_agent() -> Any:
    """Create an agent configured for web research and Excel generation."""
    from typing import Any

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope_key:
        msg = (
            "DASHSCOPE_API_KEY is not set.\n"
            "Please either:\n"
            "  1. Create a .env file in this directory with:\n"
            '     DASHSCOPE_API_KEY="sk-..."\n'
            "  2. Or export it before running:\n"
            '     export DASHSCOPE_API_KEY="sk-..."'
        )
        raise ValueError(msg)

    model = ChatOpenAI(
        model="qwen3.6-plus-2026-04-02",
        api_key=dashscope_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return create_deep_agent(
        model=model,
        tools=[web_search, generate_excel],
        system_prompt=(
            "You are an Excel Research Agent. Your workflow is:\n"
            "1. Use web_search to find relevant data for the user's question.\n"
            "2. Extract and structure the data into a clean list of dictionaries.\n"
            "3. Call generate_excel with the structured data to create the file.\n"
            "4. Confirm the file path to the user.\n\n"
            "When creating data for Excel, use clear column headers and make "
            "sure all rows have the same keys. Be concise but accurate."
        ),
    )


async def main() -> None:
    """Run the Excel Research Agent."""
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = (
            "Search for the top 5 most popular programming languages in 2024 "
            "with their approximate market share percentages, "
            "then generate an Excel file named programming_languages_2024.xlsx"
        )

    print(f"Task: {task}\n")
    agent = create_excel_agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
    )

    for msg in result["messages"]:
        print(f"\n--- {type(msg).__name__} ---")
        content = getattr(msg, "content", "")
        if content:
            print(content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                print(f"  [ToolCall] {name}({args})")
        if hasattr(msg, "name") and msg.name:
            result_text = getattr(msg, "content", "")
            print(f"  [ToolResult from {msg.name}]")
            print(f"  {result_text[:500]}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
