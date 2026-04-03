# LangChain & LangGraph
官方文档案例的本地化改造，使用OpenAIModelClass对接百炼平台/小米MiMo实现。
解决文档中对于国内模型没有直接案例的问题，复现了[LangChain官网](https://docs.langchain.com/oss/python/langchain/overview)的重点示例
## 模型
- 语言模型需要支持工具调用，案例中LangChain部分主要使用qwen-plus，部分使用qwen_max。qwen-plus免费额度足以完成LangChain部分;
    - 小米MiMo在实名认证后提供了5元免费调用额度，调用一次约0.01元，数学运算理解能力强于qwen-plus,但在结构化输出中有问题，对属性名称要求强、输出Json疑似进入检查死循环
- Embedding模型使用qwen的text-embedding-v3
## LangChain部分
按照官网文档顺序排序如下
- helloworld.py - 简单的天气查询Agent，对应[Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- helloworld_multiquestion.py - 多轮对话的天气查询Agent，具有结构化输出[Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart)
- dynamic_model.py - 动态模型切换功能，根据对话轮数选择不同模型[Dynamic Model](https://docs.langchain.com/oss/python/langchain/agents#dynamic-model)
- dynamic_tools_runtime.py - 运行时动态添加工具[Dynamic Tools](https://docs.langchain.com/oss/python/langchain/agents#dynamic-tools)
- dynamic_tools_pre_reg.py - 预注册工具，根据角色过滤工具[Dynamic Tools](https://docs.langchain.com/oss/python/langchain/agents#dynamic-tools)
- dynamic_system_prompt.py - 基于用户角色动态调整系统提示[Dynamic System Prompt](https://docs.langchain.com/oss/python/langchain/agents#dynamic-system-prompt)
- short_term_memory.py - 具有短期记忆管理的Agent系统[Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- human_in_the_loop.py - 人在环（Human-in-the-Loop）Agent系统，使用函数模拟人审批[human_in_the_loop](https://docs.langchain.com/oss/python/langchain/streaming#streaming-with-human-in-the-loop)
- streaming_sub_agent.py - 具有流式输出[Streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- structed_output.py - 具有结构化输出能力[Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)# 26-4-3发现MiMO执行时代码问题，这部分会大量消耗token，原因未知，Qwen-plus、Flash模型正常运行
- weather_mcp.py - 使用Model Context Protocol调用外部天气服务的Agent系统[Model Context Protocol (MCP)](https://docs.langchain.com/oss/python/langchain/mcp)
- custom_workflow.py - 自定义工作流系统，用于处理RAG相关查询[Multi-agent Custom workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow)
- agentic_RAG.py - Agent类型的RAG，让Agent查找LangGraph文档中的信息并回答问题[Agentic RAG](https://docs.langchain.com/oss/python/langchain/retrieval#agentic-rag)
