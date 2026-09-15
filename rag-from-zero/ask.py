from src.config import INDEX_PATH, CHUNK_META_PATH, TOP_K
from src.embedder import Embedder
from src.vector_store import FaissStore
from src.llm import chat

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
请严格依据【参考资料】回答问题。
要求：
1. 资料中没有的信息，直接回答"资料中未提及"，绝对不要编造。
2. 在回答中涉及具体事实的地方，用 [1] [2] 标注它来自哪条资料。
3. 回答简洁，先给结论再给依据。"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}"""


def build_context(hits: list[tuple[float, dict]]) -> str:
    lines = []
    for i, (score, meta) in enumerate(hits, start=1):
        loc = f"{meta['source']}" + (f" 第{meta['page']}页" if meta.get("page") else "")
        lines.append(f"[{i}] (来源：{loc}，相似度 {score:.3f})\n{meta['text']}")
    return "\n\n".join(lines)


def ask(question: str, k: int = TOP_K):
    embedder = Embedder()
    store = FaissStore(INDEX_PATH, CHUNK_META_PATH).load()

    hits = store.search(embedder.encode_query(question), k=k)
    if not hits:
        print("没有检索到任何内容，先跑 build_index.py")
        return

    context = build_context(hits)
    answer = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ])

    print("\n===== 回答 =====\n")
    print(answer)
    print("\n===== 检索到的资料 =====")
    for i, (score, meta) in enumerate(hits, start=1):
        loc = f"{meta['source']}" + (f" p{meta['page']}" if meta.get("page") else "")
        print(f"[{i}] {score:.3f} {loc}: {meta['text'][:60]}...")


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or input("请输入问题：")
    ask(q)