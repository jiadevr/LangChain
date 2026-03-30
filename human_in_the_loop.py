# [人在环模拟](https://docs.langchain.com/oss/python/langchain/streaming#streaming-with-human-in-the-loop)
from typing import Dict
from typing import List
from langchain_openai import ChatOpenAI
import os
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


checkpointer = InMemorySaver()

chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

agent = create_agent(
    model=chat_llm,
    tools=[get_weather],
    middleware=[
        # 在工具调用处自动暂停
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
    ],
    checkpointer=checkpointer,
)


# 渲染函数
def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(" [Tool call]")


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\nTool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"\nTool response: {message.content}")


def _render_interrupt(interrupt: Interrupt) -> None:
    interrupts = interrupt.value
    for request in interrupts["action_requests"]:
        print(f"\nINTERRUPT - {request['description']}")


# 统一stream处理器 chunk结构为(message,(chunk, metadata))
def handle_stream_chunk(chunk: tuple, interrupts: List) -> None:
    if not isinstance(chunk, tuple) or len(chunk) < 2:
        return

    channel, data = chunk
    if channel == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    elif channel == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
            if source == "__interrupt__":
                # 当被打断时把所有interrupt添加到列表中
                interrupts.extend(update)
                _render_interrupt(update[0])


# 决策函数，其实是模拟人决策，在Code中直接运行Input不会阻塞
def _get_interrupt_decisions(interrupt: Interrupt) -> List[Dict]:
    # 这段是个推导式，对于每个名称为Boston的执行改写
    return [
        (
            {
                "type": "edit",
                "edited_action": {
                    "name": "get_weather",
                    "args": {"city": "Boston, U.K."},
                },
            }
            if "boston" in request["description"].lower()
            else {"type": "approve"}
        )
        for request in interrupt.value["action_requests"]
    ]


# 主程序
input_message = {
    "role": "user",
    "content": "Can you look up the weather in Boston and San Francisco?",
}
config = {"configurable": {"thread_id": "some_id"}}

print("首次执行（触发HITL），调用该工具时暂停，：")
interrupts = []
for chunk in agent.stream(
    {"messages": [input_message]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    handle_stream_chunk(chunk, interrupts)

print("\n\n生成决策：")
# 进入审批函数
decisions = {}
for interrupt in interrupts:
    interrupt_id = interrupt.id  # 兼容dict/id属性
    decisions[interrupt_id] = {"decisions": _get_interrupt_decisions(interrupt)}
print(decisions)
# 依靠checkpointer保持一致性

print("\n\n恢复执行（应用决策）：")
interrupts = []  # 重置
for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    handle_stream_chunk(chunk, interrupts)

print("\n执行完成！")
