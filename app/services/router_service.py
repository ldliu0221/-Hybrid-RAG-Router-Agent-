from app.services.llm_service import LLMService


class RouterService:
    def __init__(self):
        self.llm = LLMService()

    def route(self, question: str) -> str:
        prompt = f"""
你是一个路由分类器，需要判断问题是否依赖企业知识库。

问题：
{question}

判断规则：

【RAG（必须用知识库）】
- 公司制度、报销、请假、考勤
- 内部流程、审批、系统操作
- 企业政策、规定

【LLM（不能用知识库）】
- 科普问题（如：太阳为什么发光）
- 通用知识（历史、物理、医学等）
- 与企业无关的问题

⚠️ 如果问题没有任何“公司/员工/制度”相关内容，一律判定为 LLM

输出要求：
只能输出：RAG 或 LLM
不要解释
"""

        result = self.llm.simple_chat(prompt).strip().upper()

        print("Router input:", question)
        print("Router raw result:", result)

        if "RAG" in result:
            return "rag"
        return "llm"