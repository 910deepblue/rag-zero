# -*- coding: utf-8 -*-
"""
build_index.py —— 建库入口脚本（离线跑一次）

作用：把 data/ 下的文档，走完 加载 → 切分 → 向量化 → 建库落盘 四步，
      产出 index.faiss + meta.jsonl 两个文件，供后续问答脚本检索。

运行方式（重要）：
    必须在项目根目录 E:\rag-from-zero 下执行
        (.venv) E:\rag-from-zero>python build_index.py
    因为导入用的是【绝对导入】(from src.xxx)，把工作目录切到 src 里再跑会直接 ImportError。
    （对比 embedder.py 内的 `from .config import` 是包内【相对导入】，两种写法不能混用同一套目录假设）
"""

# 所有可变参数(DATA_DIR / 路径 / 切块大小与重叠)统一从 config 拿，
# 脚本本身零硬编码 —— 调参只改 config.py，不用动流水线代码。
from src.config import (DATA_DIR, INDEX_PATH, CHUNK_META_PATH,
                        CHUNK_SIZE, CHUNK_OVERLAP)
from src.loader import load_dir          # [第1步] 文档 → 文本单元
from src.splitter import split_units     # [第2步] 长文本 → 小块(这里没有长文本，用的是"单元→块")
from src.embedder import Embedder        # [第3步] 文本 → 向量
from src.vector_store import FaissStore  # [第4步] 向量+原文 → 索引与 jsonl


def main():
    # 用 "="*50 分隔 + [n/4] 编号，本质上是个极简进度条。
    # 长跑任务这么写很值：出问题时一眼能看出卡在第几步。
    print("=" * 50, "\n[1/4] 解析文档")

    # 递归读取 DATA_DIR，按后缀分派给对应 loader，返回 [{text, source, page}, ...]
    units = load_dir(DATA_DIR)

    # 早退保护：没有文档就别往下走，否则会建出一个 0 条的索引，
    # 后面检索时 score 全空、报错信息还很难懂。
    # （小优化：改成 sys.exit(1) 更好，能让 shell/CI 知道这次是【失败】而不是成功）
    if not units:
        print("data/ 目录里没有可用文档，先放几个进去。")
        return

    print("=" * 50, "\n[2/4] 切分")
    # 切块：CHUNK_SIZE 是每块的目标长度，CHUNK_OVERLAP 是相邻块的重叠量
    # （重叠是为了防止答案正好被切在两个块的分界上，切一刀就丢了上下文）
    chunks = split_units(units, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"共 {len(chunks)} 块")

    print("=" * 50, "\n[3/4] 向量化")
    # 模型只加载一次并复用：SentenceTransformer 初始化要读几百 MB 权重，很慢，
    # 千万别写成循环里每次 new 一个 Embedder。
    # （本机是 CPU 版 torch，没有 CUDA，这一步通常是最慢的环节）
    embedder = Embedder()

    # 从块字典里只抽出 text 字段，对齐 encode_docs(list[str]) 的入参签名。
    # ⚠️ 关键：这里必须保持顺序不变！下面 vectors 的第 i 行 ⇄ chunks[i]，
    #        一旦打乱，检索就会返回错的原文（build 里的 assert 只校验数量，不校验顺序）。
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode_docs(texts)

    print("=" * 50, "\n[4/4] 建库落盘")
    # 一行做完"构造 → 建索引 → 写盘"：FaissStore.build() 内部会调用 save()。
    # 这里不给对象取名字，是因为后续流程用不到它 —— 只在建库这一刻存在。
    # 注意 save() 用的是 open(..., "w")：每次都是【全量覆盖重建】。
    # 加了新文档要重新跑本脚本；IndexFlat 是暴力索引，重建本身很快(几万条几秒级)。
    FaissStore(INDEX_PATH, CHUNK_META_PATH).build(vectors, chunks)

    print("完成。")


# 守卫：只有"直接运行本文件"才执行 main()。
# 被别人 import 时(比如写测试、或做增量脚本)不会自动建库 —— 这是个好习惯，别删。
if __name__ == "__main__":
    main()
