import numpy as np
from sentence_transformers import SentenceTransformer

# 1) 准备"知识库"（真实项目里这里是从文档切出来的块）
docs = [
    "RAG 是检索增强生成，先用检索找到相关资料，再让大模型基于资料回答。",
    "向量数据库用于存储文本的向量表示，并支持相似度检索。",
    "FAISS 是 Meta 开源的向量检索库，支持十亿级向量的快速搜索。",
    "北京是中国的首都，常住人口超过两千万。",
    "BGE 是智源研究院发布的中文向量模型系列。",
]

# 2) 加载向量模型（走本地目录，不联网）
model = SentenceTransformer(r"D:\DevEnv\ai-cache\models\bge-small-zh-v1.5")

# 3) 把所有文档编码成向量。normalize_embeddings=True 后，内积 = 余弦相似度
doc_vecs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)

# 4) 提问
question = "向量检索用什么库比较好？"
q_vec = model.encode([question], normalize_embeddings=True)[0]

# 5) 相似度 = 矩阵乘法，一步算完所有文档
scores = doc_vecs @ q_vec

# 6) 取 Top-2
top_idx = np.argsort(-scores)[:2]
for i in top_idx:
    print(f"{scores[i]:.4f}  {docs[i]}")