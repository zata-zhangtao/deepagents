# Excel Research Agent

基于 Deep Agents 的网页搜索 + Excel 生成应用。

## 功能

- **网页搜索**：使用 DuckDuckGo 搜索，**不需要 API Key**
- **数据结构化**：自动提取并整理搜索结果
- **Excel 生成**：生成带表头样式、自动列宽的 `.xlsx` 文件

## 安装

```bash
cd examples/excel-research-agent
uv sync
```

## 运行

```bash
export DASHSCOPE_API_KEY="sk-..."

# 使用默认任务
uv run python agent.py

# 自定义任务
uv run python agent.py "2024年全球智能手机市场份额"

# 指定文件名
uv run python agent.py "搜索2024年AI融资最多的10家公司并生成excel文件"
```

## 示例输出

Agent 会自动：
1. 调用 `web_search` 搜索网页
2. 整理数据成结构化表格
3. 调用 `generate_excel` 生成 `*.xlsx` 文件
4. 告诉你文件保存在哪里

## 技术栈

- **Deep Agents SDK** — Agent 编排
- **DuckDuckGo Search** — 免费网页搜索
- **OpenPyXL** — Excel 文件生成
- **DashScope (通义千问)** — LLM
