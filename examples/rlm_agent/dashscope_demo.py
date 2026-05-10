"""Deep Agents + DashScope (通义千问) 示例

展示 deepagents 内置工具的使用：todo 管理、文件读写等。
"""
import os

from langchain_community.chat_models.tongyi import ChatTongyi
from deepagents import create_deep_agent

os.environ["DASHSCOPE_API_KEY"] = "sk-1e631a8dda5f46cba0ef26a6bf1dcbd1"

model = ChatTongyi(model_name="qwen-plus")
agent = create_deep_agent(model=model)

task = (
    "Please do the following:\n"
    "1. Create a todo list for learning Python (3 items).\n"
    "2. Write a hello_world.py file that prints 'Hello from Deep Agents!'.\n"
    "3. Read back the file to confirm it was written correctly."
)

print("=" * 60)
print("Task:", task)
print("=" * 60)

result = agent.invoke({"messages": [{"role": "user", "content": task}]})

for msg in result["messages"]:
    print(f"\n--- {type(msg).__name__} ---")
    content = getattr(msg, "content", "")
    if content:
        print(content)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  [ToolCall] {tc.get('name')}({tc.get('args')})")
    if hasattr(msg, "name") and msg.name:
        print(f"  [ToolResult from {msg.name}]")
        print(f"  {getattr(msg, 'content', '')[:500]}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
