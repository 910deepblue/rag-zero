# -*- coding: utf-8 -*-
"""
store.py —— 极简本地向量库

设计取舍：
  - FAISS 只存向量（它本身不存原文），原文和来源另存成 jsonl；
  - 两边用【位置顺序】对齐：第 N 个向量 ←→ meta.jsonl 第 N 行（行号即 id，天然免维护）；
  - jsonl 而非 json：一行一条，追加/流式读都方便，中途写坏也只坏一行。
"""

import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """极简本地向量库：FAISS 存向量，jsonl 存原文与来源（行号即 id）"""

    def __init__(self, index_path: Path, meta_path: Path):
        # Path() 再包一层：调用方传 str 还是 Path 都能吃，属于防御性写法
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        # 懒加载：先置空，等到真正 search 时才 load()。
        # 好处：只负责建库的流程不需要把索引读进内存。
        # 注意 `faiss.Index | None` 只是【类型注解】，运行时不检查；
        # 这种 X | Y 写法要求 Python >= 3.10（本机 3.12.7，没问题）。
        self.index: faiss.Index | None = None   # 这就是所谓的懒加载。
        self.metas: list[dict] = []          # 与 index 内的向量一一对应

    # ---------- 写入 ----------

    def build(self, vectors: np.ndarray, metas: list[dict]) -> None:
        """从零建库：向量矩阵 + 对应的元数据列表，建完立即落盘"""
        # 数量必须一致，否则位置对齐的约定直接崩掉 —— 属于"早失败优于错结果"
        assert vectors.shape[0] == len(metas), "向量数与元数据数不一致"

        dim = vectors.shape[1]               # 向量维度，必须与后续查询向量一致

        # 向量已归一化 -> 用内积索引，等价于余弦相似度
        # IndexFlatIP = 暴力精确检索（不建图、不聚类），库小(万级以下)时最快也最准；
        # 数据量上去后可以换成 IndexIVFFlat / HNSW 做近似检索。
        self.index = faiss.IndexFlatIP(dim)

        # add 的输入必须是 float32（上游 embedder 里 astype("float32") 正是为此），
        # 且要求内存连续；shape 必须是 (n, dim)。
        self.index.add(vectors)

        self.metas = metas
        self.save()                          # 建完立刻持久化，避免进程退出丢结果

        print(f"[建库] {len(metas)} 条, 维度 {dim} -> {self.index_path}")

    def save(self) -> None:
        """落盘：索引写成 faiss 二进制，元数据写成 jsonl"""
        # 首次运行 index 目录不存在 -> parents=True 递归建目录，exist_ok=True 已存在不报错
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # faiss 写入接口只接受 str 路径，不吃 Path 对象（与 python-docx 同类毛病）
        faiss.write_index(self.index, str(self.index_path))

        # 用 open(..., "w") + 手动逐行写，是为了精确控制"一行一条"这个格式；
        # 若某个 meta 里含换行，json.dumps 会转义成 \n，所以永远只占一行 —— 行号对应关系安全。
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                # ensure_ascii=False：中文原样写入，不变成 \uXXXX，方便肉眼查库
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- 读取 ----------

    def load(self) -> "FaissStore":
        """从磁盘恢复索引与元数据；返回 self 以便链式调用：FaissStore(p, m).load()"""
        self.index = faiss.read_index(str(self.index_path))

        # splitlines() 按行切；if line.strip() 跳过末尾空行/空白行，
        # 保证 metas 的下标与 FAISS 返回的索引编号严格一致（差一行就全错位）。
        self.metas = [
            json.loads(line)
            for line in self.meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return self

    # ---------- 查询 ----------

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        """返回 top-k 命中，每项是 (相似度分数, 元数据)"""
        if self.index is None:               # 没用过就没加载过 -> 自动补加载
            self.load()

        # reshape(1, -1)：faiss.search 要求二维输入 (n_queries, dim)；
        # 查询向量是 (dim,) 的一维，这里补一维变成"1 条查询"。
        # asarray(dtype="float32")：兜底转换，防止上游漏了 float32 导致 faiss 直接报错。
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)

        # 返回两个 (1, k) 的数组：scores 是内积分数，idx 是命中的向量下标(即 metas 的下标)
        scores, idx = self.index.search(q, k)

        hits = []
        for score, i in zip(scores[0], idx[0]):   # [0] 取第一行（我们只查了一条）
            if i == -1:                      # 库里不足 k 条时，faiss 用 -1 填充空位，必须跳过
                continue
            # score 是 float32，转成 Python float 更干净(方便 json 序列化/打印)；
            # 因为两侧都做了归一化，这个内积值就等于余弦相似度，范围 [-1, 1]。
            hits.append((float(score), self.metas[i]))
        return hits
