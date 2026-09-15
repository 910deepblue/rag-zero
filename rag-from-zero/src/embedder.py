# -*- coding: utf-8 -*-
"""
embedder.py —— 向量化模块

职责：把文本(文档块 / 用户查询)转成定长向量，供 faiss 建索引与检索。
说明：模型加载很慢(几百 MB 权重要读盘)，所以封装成类、实例只建一次，
      全流程复用同一个 self.model，不要每次编码都 new 一个。
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# 相对导入：说明本文件是 rag 包内的模块，必须通过 `python -m rag.xxx` 或包内导入运行；
# 直接 `python embedder.py` 会因为找不到父包而报 ImportError。
# EMBEDDING_MODEL：模型名或本地模型目录(本机"本地模型存在: True 文件数: 10"就是走本地目录，离线可用)
# QUERY_INSTRUCTION：BGE 官方要求加在查询侧的前缀指令
from .config import EMBEDDING_MODEL, QUERY_INSTRUCTION


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        # 传入 device：本机 torch 是 CPU 版(cuda: False)，所以 "cpu" 是正确的默认值；
        # 哪天换上 CUDA 版 torch，把这里改成 "cuda" 就能显著提速(批量编码是这类项目的耗时大头)。
        self.model = SentenceTransformer(model_name, device=device)

        # ⚠️ 【修订 7】sentence-transformers 6.x 把旧名 get_sentence_embedding_dimension()
        #    改成了 get_embedding_dimension()。旧名还能用，但每次都会打 FutureWarning：
        #      FutureWarning: The `get_sentence_embedding_dimension` method has been renamed
        #                     to `get_embedding_dimension`.
        #    下面这样写，新旧版本都能跑（本机装的是 6.0.1，走新名字这条路）。
        #
        # 拆解这段兼容写法：
        #   getattr(模型, "新名", None)  → 新版本返回【绑定方法】(真值)；老版本返回 None(假值)
        #   A or B                      → A 为真用 A，否则回退到旧名，恰好实现"新优先、旧兜底"
        #   注意拿到的是"方法"而不是"结果"，所以后面还要再调一次 get_dim()
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()   # 向量维度，建 faiss 索引时要用(如 IndexFlatIP(dim))

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """把一批文档块编码成向量矩阵，形状 (条数, dim)，dtype=float32"""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # 归一化后内积 == 余弦相似度
            show_progress_bar=True,      # 批量编码耗时长，给个进度条；接服务/   写日志时建议关掉(会刷 stderr)
            convert_to_numpy=True,       # 直接产出 np.ndarray，省掉自己转换
        )
        # faiss 只接受 float32。若上游给了 float64，不转会在建索引时报 dtype 不匹配；
        # 就算侥幸能跑，也会被内部再拷一份，白白翻倍内存。
        return vecs.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        """把单条用户查询编码成向量，形状 (dim,)，dtype=float32"""
        # 检索侧加指令前缀，BGE 官方推荐做法，召回更准
        # ⚠️ 关键：前缀【只能加在查询侧】。文档侧(encode_docs)绝对不能加，
        #    否则两侧不在同一个语义空间，召回率会明显掉。
        vec = self.model.encode(
            [QUERY_INSTRUCTION + text],      # 即使只有一条也要包成 list：encode() 返回二维数组
            normalize_embeddings=True,       # 必须与 encode_docs 保持一致
            convert_to_numpy=True,
        )
        # 取 [0] 从 (1, dim) 降成 (dim,)，因为查询侧通常按单个向量用(与索引内积算相似度)
        return vec[0].astype("float32")
