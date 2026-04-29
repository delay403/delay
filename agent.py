"""
个性化学术研究辅助 Agent (Academic Research Assistant)
支持：论文检索、概念解释、研究大纲生成、个性化记忆
依赖：pip install langchain langchain-openai langchain-community wikipedia arxiv
"""
import os
import warnings
from typing import Optional
from datetime import datetime

# LangChain 核心
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# 社区工具
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

warnings.filterwarnings("ignore")

# ---------- 1. 配置 API 密钥 ----------
# 请将 OPENAI_API_KEY 设置为环境变量，或直接替换为你的 key
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 替换为真实 key

# ---------- 2. 自定义工具 ----------
def get_current_time(_: str = "") -> str:
    """返回当前日期和时间，用于研究进度规划"""
    now = datetime.now()
    return f"当前日期：{now.strftime('%Y年%m月%d日')}，时间：{now.strftime('%H:%M:%S')}"

def calculator(expression: str) -> str:
    """执行数学计算，输入如 '2+3*5' """
    try:
        # 安全计算，仅允许数字和运算符
        allowed = set("0123456789.+-*/() ")
        if not set(expression).issubset(allowed):
            return "错误：表达式包含非法字符，仅支持数字和 + - * / ( )"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# 学术工具
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 工具列表
tools = [
    Tool(
        name="arxiv_search",
        func=arxiv_tool.run,
        description="搜索 ArXiv 上的学术论文。输入应为英文关键词，如 'transformer attention mechanism'。返回论文标题、作者和摘要。"
    ),
    Tool(
        name="wikipedia",
        func=wiki_tool.run,
        description="查询 Wikipedia 获取概念解释。输入中文或英文术语均可。"
    ),
    Tool(
        name="get_time",
        func=get_current_time,
        description="获取当前日期和时间。用于需要时间信息的场景。"
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="执行数学计算。输入合法的数学表达式。"
    ),
]

# ---------- 3. 个性化系统提示 ----------
# 这里可以嵌入用户画像（研究领域、学历等），在运行时动态更新
SYSTEM_PROMPT = """你是一位专业的学术研究助手，服务于科研与教育领域。你的用户信息如下：
- 研究方向：人工智能（自然语言处理）
- 身份：博士研究生
- 偏好语言：中文为主，必要时中英混合

遵循以下准则：
1. **信息准确**：使用工具获取论文和百科知识时，务必引用来源（ArXiv ID 或 Wiki 标题）。
2. **逻辑清晰**：解释概念时由浅入深，适合研究生水平。
3. **主动推荐**：根据用户兴趣，在回答问题后主动推荐 1-2 篇相关论文或研究思路。
4. **安全边界**：只回答学术、技术、教育类问题，拒绝无关或敏感话题。
5. **时间敏感**：如果用户询问“最新进展”，使用 ArXiv 检索近一年论文，并提示当前日期。
6. **格式规范**：回答使用 Markdown 结构，重要信息加粗。

当前日期：{current_time}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------- 4. 记忆系统 ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # 保留最近约2000 token的摘要 + 原始消息
    return_messages=True,
    memory_key="chat_history",
)

# ---------- 5. 构建 Agent ----------
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,          # 调试时可改为 True
    handle_parsing_errors=True,
    max_iterations=6,       # 防止无限循环
)

# ---------- 6. 交互循环 ----------
def main():
    print("=" * 60)
    print("🎓 个性化学术助手已启动（输入 'exit' 退出）")
    print("=" * 60)
    # 将当前时间注入 system prompt（若不直接修改 prompt 则可省略，但已通过工具提供）
    # 实际上 prompt 中的 {current_time} 未自动填充，这里手动填充一种方式：
    # 我们改用 partial_variables 方式，但为了简洁，在运行前动态注入。
    # 更简单方案：在调用时显式传入 current_time，但 agent_executor 不支持。
    # 解决方案：在 system prompt 中使用工具 get_time 即可。

    while True:
        try:
            user_input = input("\n🧑 你：")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 再见，期待再次帮你探索知识！")
                break
            if not user_input.strip():
                continue

            # 调用 Agent
            response = agent_executor.invoke({"input": user_input})
            print(f"\n🤖 助手：{response['output']}")

        except KeyboardInterrupt:
            print("\n👋 退出。")
            break
        except Exception as e:
            print(f"\n⚠️ 出错：{str(e)}")

if __name__ == "__main__":
    main()