#[Agentic RAG - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/retrieval#agentic-rag)
#使用Agent检索LangGraph文档示例
import os
import sys
import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from markdownify import markdownify




ALLOWED_DOMAINS = ["https://langchain-ai.github.io/","https://docs.langchain.com/oss/python/langgraph/"]
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt'

chat_llm= ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

@tool
def fetch_document(url: str) -> str:
    """Fetch and convert document from a given URL."""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response=requests.get(url,timeout=10.0)
    response.raise_for_status()
    print("______fetch_data_________")
    print(response.text)
    return markdownify(response.text)


#https://langchain-ai.github.io/langgraph/llms.txt经常访问不到，直接列出
llms_txt_content="""
# LangGraph

LangGraph documentation has moved to docs.langchain.com.

## Overview

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview): Introduction to LangGraph, a library for building stateful, multi-actor applications with LLMs.
- [Why LangGraph?](https://docs.langchain.com/oss/python/langgraph/why-langgraph): Motivation for LangGraph and its key features.

## Core Concepts

- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): Learn how to define state, create nodes, and connect them with edges.
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming): Stream outputs from your graph for better UX.
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence): Add memory and checkpointing to your graphs.
- [Add Memory](https://docs.langchain.com/oss/python/langgraph/add-memory): Implement short-term and long-term memory.
- [Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents): Build agents and workflows with LangGraph.

## How-To Guides

- [Use Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs): Compose graphs using subgraphs.
- [Observability](https://docs.langchain.com/oss/python/langgraph/observability): Add tracing and debugging to your graphs.
- [Common Errors](https://docs.langchain.com/oss/python/langgraph/common-errors): Troubleshoot common LangGraph errors.

## Tutorials

- [Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag): Build an agentic RAG system with LangGraph.
- [SQL Agent](https://docs.langchain.com/oss/python/langgraph/sql-agent): Create a SQL agent with LangGraph.

## Reference

- [API Reference](https://reference.langchain.com/python/langgraph/): Complete API documentation for LangGraph.

## LangGraph Platform

For deploying LangGraph applications in production, see the [LangSmith documentation](https://docs.langchain.com/langsmith/agent-server).
"""


# System prompt for the agent
system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about LangGraph and related tools.

Instructions:

1. If a user asks a question you're unsure about—or one that likely involves API usage,
   behavior, or configuration—you MUST use the `fetch_documentation` tool to consult the relevant docs.
2. When citing documentation, summarize clearly and include relevant context from the content.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best expert understanding.

You can access official documentation from the following approved sources:

{llms_txt_content}

You MUST consult the documentation to get up to date documentation
before answering a user's question about LangGraph.

Your answers should be clear, concise, and technically accurate.
"""

agent = create_agent(
    model=chat_llm,
    tools=[
        fetch_document
    ],
    name="Agentic_RAG",
    system_prompt=system_prompt,
)
# 默认访问的网页(https://docs.langchain.com/oss/python/langgraph/prebuilt-agents)404了
response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent,"
            "which should be able to look up stock pricing information."
        ))
    ]
})
print("\n\n______response_________\n\n")
print(response['messages'][-1].content)