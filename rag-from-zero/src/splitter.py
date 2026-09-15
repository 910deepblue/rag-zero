import re

# 切分优先级：先按段落，再按句子，最后才硬切
_SPLIT_PATTERN = re.compile(r"(?<=[。！？；!?;\n])")


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """按语义边界切块，并给相邻块加重叠，避免切断上下文。"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    sentences = [s for s in _SPLIT_PATTERN.split(text) if s.strip()]

    # 第一轮：贪心拼接，尽量接近 chunk_size 才断开
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            # 单个句子就超长 -> 直接硬切，还要留 overlap
            while len(s) > chunk_size:
                chunks.append(s[:chunk_size])
                s = s[chunk_size - overlap:]
            cur = s
    if cur:
        chunks.append(cur)

    # 第二轮：把上一块的尾巴接到当前块前面，形成重叠
    final = []
    for i, c in enumerate(chunks):
        if i == 0 or overlap <= 0:
            final.append(c.strip())
        else:
            final.append((chunks[i - 1][-overlap:] + c).strip())
    return [c for c in final if c]


def split_units(units: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """对 load_dir 的输出做切分，给每块补上全局 id 和块内序号"""
    out = []
    for u in units:
        for j, piece in enumerate(split_text(u["text"], chunk_size, overlap)):
            out.append({
                "id": len(out),
                "text": piece,
                "source": u["source"],
                "page": u["page"],
                "part": j,          # 同一页/文件的第几块
            })
    return out