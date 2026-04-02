import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from openai import OpenAI
from app.core.config import settings


def main():
    print("=== DashScope Config Check ===")
    print("ROOT_DIR:", ROOT_DIR)
    print("base_url:", settings.dashscope_base_url)
    print("model:", settings.dashscope_model)
    print("api_key exists:", bool(settings.dashscope_api_key))

    if not settings.dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY 未读取到，请检查 .env 文件内容")

    client = OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )

    completion = client.chat.completions.create(
        model=settings.dashscope_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ],
        temperature=0.2,
    )

    print("\n=== Model Response ===")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()