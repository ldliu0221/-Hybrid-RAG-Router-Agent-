# 🚀 Enterprise RAG Agent QA

> 基于 **Hybrid RAG + Router Agent** 的企业知识问答系统，支持动态路由、混合检索、重排序与自动评测。

---

## 📌 项目简介

本项目面向企业制度与流程问答场景，构建了一个完整的智能问答系统。系统能够根据问题语义自动选择 **RAG（检索增强生成）** 或 **通用 LLM** 路径，并通过混合检索（Hybrid Retrieval）、重排序（Rerank）与评测机制（LLM-as-a-Judge），提升问答准确率与稳定性。

---

## ✨ 核心能力

- 📚 多格式文档解析与知识入库（PDF / Markdown / TXT）
- 🔍 Hybrid Retrieval（Embedding + BM25）
- ✍️ Query Rewrite（问题改写）
- 🧠 Cross-Encoder Rerank（精排）
- 🤖 Router Agent（动态路由）
- 📊 自动评测系统（LLM-as-a-Judge）
- ⚙️ 自动调参系统（Retrieval Tuning）

---

## 🏗️ 系统架构

```text
User Query
    ↓
Router Agent
 ├── RAG Pipeline
 │    ├── Query Rewrite
 │    ├── Dense Retrieval (Embedding)
 │    ├── Sparse Retrieval (BM25)
 │    ├── Score Fusion
 │    ├── Rerank
 │    └── LLM Answer + Citation
 │
 └── LLM Direct Answer
    ↓
Evaluation (LLM-as-a-Judge)
```

---

## 📂 项目结构

```text
.
├── app/
│   ├── api/              # FastAPI 路由
│   ├── core/             # 配置管理
│   ├── models/           # 数据结构
│   ├── services/         # 核心逻辑（RAG / Eval / Agent）
│   ├── agent/            # Router / Rewrite
│   ├── eval/             # 数据集 & 评测结果
│   └── main.py
│
├── scripts/              # 工具脚本
│   ├── run_eval.py
│   ├── tune_retrieval.py
│   ├── load_demo_docs.py
│   └── check_qdrant.py
│
├── data/                 # 本地数据（不提交运行态数据）
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/yourname/enterprise-rag-agent-qa.git
cd enterprise-rag-agent-qa
```

### 2️⃣ 创建环境

```bash
conda create -n rag-agent python=3.10
conda activate rag-agent
```

或：

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 4️⃣ 配置环境变量

复制 `.env.example` 为 `.env`，并填写：

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

QDRANT_COLLECTION=enterprise_knowledge
```

### 5️⃣ 启动服务

```bash
uvicorn app.main:app --reload
```

访问：

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 示例接口

### 📥 文档入库

```text
POST /ingest
```

支持上传 `.txt / .md / .pdf` 文档。

### ❓ 问答接口

```json
POST /query
{
  "question": "报销多久？",
  "use_agent": true,
  "top_k": 5
}
```

示例返回：

```json
{
  "answer": "员工完成消费后，应在五个工作日内提交报销单。",
  "citations": [
    {
      "document_id": "xxx",
      "filename": "demo_reimburse.txt",
      "chunk_id": "xxx_0",
      "text": "公司的报销流程如下：员工完成消费后，应在五个工作日内提交报销单，并附上发票和支付凭证，由财务部门审核后打款。",
      "score": 0.89
    }
  ],
  "route": "rag"
}
```

### 📊 评测接口

```json
POST /eval
{
  "question": "报销多久？",
  "ground_truth": "员工完成消费后，应在五个工作日内提交报销单。",
  "use_agent": true,
  "top_k": 5
}
```

示例返回：

```json
{
  "question": "报销多久？",
  "ground_truth": "员工完成消费后，应在五个工作日内提交报销单。",
  "prediction": "员工完成消费后，应在五个工作日内提交报销单。",
  "score": 1.0,
  "reason": "模型回答与标准答案完全一致，信息完整准确。",
  "route": "rag"
}
```

---

## 📈 评测结果

```json
{
  "total_samples": 8,
  "average_score": 1.0,
  "route_stats": {
    "rag": {
      "count": 6,
      "average_score": 1.0
    },
    "llm": {
      "count": 2,
      "average_score": 1.0
    }
  }
}
```

---

## ⚙️ 自动调参结果

```json
{
  "top_k": 5,
  "dense_weight": 0.7,
  "sparse_weight": 0.3,
  "enable_rewrite": true,
  "enable_rerank": true,
  "average_score": 1.0
}
```

---

## 🧠 技术亮点

- Hybrid Retrieval：Embedding + BM25 混合召回
- 中文 BM25：结合 `jieba` 分词优化中文检索
- Cross-Encoder Rerank：提升上下文排序质量
- Router Agent：动态判断走 RAG 或 LLM
- LLM-as-a-Judge：自动评估问答质量
- Retrieval Tuning：自动调参优化检索效果

---

## 🚧 后续优化方向

- 多轮对话记忆（Memory）
- Redis / 分布式缓存
- Qdrant Server / Docker 部署
- Tool Calling Agent（数据库 / API）
- 前端 Chat UI

---

## ⭐ 项目总结

本项目围绕 RAG 构建了完整工程闭环，包括检索优化、Agent 路由与自动评测，是一个具备实际生产潜力的 LLM 应用系统。
