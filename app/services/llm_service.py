from openai import OpenAI
from app.core.config import settings


class LLMService:
    def __init__(self):
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置，请检查 .env 文件")

        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self.model = settings.dashscope_model

    def simple_chat(self, prompt: str) -> str:
        """
        通用对话接口：
        - Router 判断
        - 普通 LLM 回答
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨、简洁的智能助手。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""

    def answer_with_context(self, question: str, contexts: list[str]) -> str:
        """
        基于检索上下文回答问题。
        严格限制回答范围，避免模型自由发挥。
        """
        context_text = "\n\n".join(
            [f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts)]
        )

        prompt = f"""
你是企业知识库问答助手。

请严格根据“上下文”回答问题，禁止补充上下文中没有的信息。

问题：
{question}

上下文：
{context_text}

回答要求：
1. 只能使用上下文中的信息
2. 不允许扩展、推测或补充额外内容
3. 如果上下文中没有答案，直接回答：无法从资料中确认
4. 回答要简洁明确
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的企业知识问答助手，必须严格依据检索到的上下文回答。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )

        return completion.choices[0].message.content or ""

    def rewrite_query(self, question: str) -> str:
        """
        将用户问题改写为更适合知识库检索的查询表达。
        """
        prompt = f"""
请将下面的用户问题改写为更适合知识库检索的查询语句。

要求：
1. 保留原始语义
2. 补充关键实体、业务动作、约束条件
3. 适合用于向量检索和关键词检索
4. 只输出改写后的查询，不要解释

用户问题：
{question}
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个检索优化助手，擅长把用户问题改写为适合知识库检索的表达。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )

        return (completion.choices[0].message.content or "").strip()