# [Build a custom RAG agent with LangGraph - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
# LangChain\custom_workflow.py演示了使用本地文档进行RAG的流程,这个案例读取Web网页、涉及后评估和多分支
# [LangChain+WebBaseLoader实现大模型基于网页内容的问答系统 - 墨天轮](https://www.modb.pro/db/1922217599035781120)
from langchain_core.messages import convert_to_messages
from langchain_community.document_loaders import WebBaseLoader
import bs4
import os
from typing import Literal, TypedDict, Annotated
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, chat
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import ToolNode, tools_condition

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
# 这个读法会破坏MD文件本身的格式,基本上属于把网页原样粘出来，可能需要一些更好的网页读取包,在rag案例中使用了markdownify包
# 读取文档，原始方法近乎是拷贝整个网页，会产生大量空行等信息,使用bs4抓取特定区域
web_loader = WebBaseLoader(web_path=urls,bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content"))))
try:
    docs=web_loader.load()
    
    print(f"Loaded {len(docs)} ,length{[len(doc.page_content) for doc in docs]}")
except:
    print("Error loading documents from web loader")

# 文本切分,根据OpenAI的titoken编码进行切分，是基于token而不是字符的可以保证切分结果不超过模型输入上限，切分结果大概是不计空格300词
# 输出形式为[Document(metadata={},page_content=' '),]
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)
doc_splits=text_splitter.split_documents(docs)

# Embedding模型
embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v3",
    # 如果API要求Str明文必须关闭这项，否则报格式错
    check_embedding_ctx_length=False,
    # Qwen的Embedding批次大小为10，单批次最大Token数量为8192
    chunk_size=10
)   

vectorstore = InMemoryVectorStore(embedding=embedding_model)
vectorstore.add_documents(doc_splits)

# 召回器定义
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

@tool
def retrieve_blog_posts(query:str) -> str:
    """
    从向量数据库中召回
    """
    docs= retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool=retrieve_blog_posts

# response=retriever_tool.invoke({"query": "types of reward hacking"})

# #print(response)
chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-flash",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)
# 用于对召回文档进行评估
grade_model=chat_llm.model_copy()

# Chat模型绑定召回工具
llm_with_tools=chat_llm.bind_tools([retriever_tool])

# 调用LLM，让LLM决定是否查询文档，这个和custom_workflow.py中的差别在于后者必须先查询文档、没有对召回文档进行再评估
def generate_query_or_response(state: MessageState):
    response=(llm_with_tools.invoke(state["messages"]))
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocument(TypedDict):
    binary_score: Annotated[str,...,"Relevance score: 'yes' if relevant, or 'no' if not relevant"]

def grade_document(state: MessageState) ->Literal["generate_answer","rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response=(grade_model.with_structured_output(GradeDocument).invoke([{"role": "user", "content": prompt}]))
    score = response['binary_score']

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# 改写查询，直接要求对用户问题进行改写
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)
def rewrite_question(state: MessageState):
    messages=state["messages"]
    question=messages[0].content
    prompt=REWRITE_PROMPT.format(question=question)
    response=(chat_llm.invoke([{"role": "user", "content": prompt}]))
    return {"messages": [HumanMessage(content=response.content)]}

# 生成答案
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessageState):
    """Generate an answer."""
    question = state["messages"][0].content
    # 召回结果
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = chat_llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


graph_builder=StateGraph(MessageState)
graph_builder.add_node(generate_query_or_response)
graph_builder.add_node("retrieve",ToolNode([retriever_tool]))
graph_builder.add_node(rewrite_question)
graph_builder.add_node(generate_answer)

graph_builder.add_edge(START,"generate_query_or_response")
# 两种条件边写法
# 如果不进行召回则直接返回答案
graph_builder.add_conditional_edges("generate_query_or_response",tools_condition,{"tools":"retrieve",END:END},)
# 根据召回文档评估情况判断是否进行改写
graph_builder.add_conditional_edges("retrieve",grade_document)

graph_builder.add_edge("rewrite_question","generate_query_or_response")
graph_builder.add_edge("generate_answer",END)

rag_workflow=graph_builder.compile()

for chunk in rag_workflow.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
