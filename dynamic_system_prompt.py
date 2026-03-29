from langchain.agents.middleware.types import ModelRequest
from dataclasses import dataclass
import os
from langchain.agents.middleware.types import dynamic_prompt
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


@dataclass
class CustomContext:
    """Custom runtime context schema."""

    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    # 这个用get拿不到，没有对应的方法
    user_role = request.runtime.context.user_role
    base_prompt = "你是一个机器学习学习助手，帮助用户回答机器学习相关的问题"

    if user_role == "expert":
        return f"{base_prompt}，提供尽量详细的技术细节"
    elif user_role == "beginner":
        return f"{base_prompt}，简明易懂，尽量不使用专业名词"

    return base_prompt


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
    middleware=[
        user_role_prompt,
    ],
    context_schema=CustomContext,
)

print("-----------------as beginner----------------")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "并帮我解释机器学习和神经网络的关系"}]},
    context=CustomContext(user_role="beginner"),
)
print(response)

print("-----------------as expert----------------")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "并帮我解释机器学习和神经网络的关系"}]},
    context=CustomContext(user_role="expert"),
)
print(response)
