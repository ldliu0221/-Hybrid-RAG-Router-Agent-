import json
from app.services.answer_service import AnswerService
from app.services.router_service import RouterService
from app.services.llm_service import LLMService


class EvalService:
    def __init__(self):
        self.router_service = RouterService()
        self.llm_service = LLMService()

    def run_single_eval(
        self,
        question: str,
        ground_truth: str,
        use_agent: bool = True,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        enable_rewrite: bool = True,
        enable_rerank: bool = True,
    ):
        answer_service = AnswerService(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            enable_rewrite=enable_rewrite,
            enable_rerank=enable_rerank,
        )

        if use_agent:
            route = self.router_service.route(question)
        else:
            route = "rag"

        if route == "rag":
            prediction, _ = answer_service.answer(
                question=question,
                top_k=top_k,
            )
        else:
            prediction = self.llm_service.simple_chat(question)

        judge_prompt = f"""
你是一个问答评测裁判。请根据“问题、标准答案、模型回答”进行评分。

评分标准：
- 1.0：回答与标准答案语义完全一致，信息完整
- 0.8：回答基本正确，只有轻微表达差异
- 0.6：回答部分正确，但不完整
- 0.3：回答有明显偏差
- 0.0：回答错误或无关

要求：
1. 只基于语义一致性评分
2. 不要因为措辞不同就扣分
3. 输出必须为 JSON
4. JSON 格式必须如下：
{{"score": 0.0, "reason": "你的简短评语"}}

问题：
{question}

标准答案：
{ground_truth}

模型回答：
{prediction}
"""

        judge_raw = self.llm_service.simple_chat(judge_prompt)

        score = 0.0
        reason = "评测结果解析失败"

        try:
            cleaned = judge_raw.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            score = float(result.get("score", 0.0))
            reason = str(result.get("reason", "无评语"))
        except Exception:
            reason = f"评测结果解析失败，原始输出：{judge_raw}"

        return {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "score": score,
            "reason": reason,
            "route": route,
            "config": {
                "top_k": top_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "enable_rewrite": enable_rewrite,
                "enable_rerank": enable_rerank,
            }
        }