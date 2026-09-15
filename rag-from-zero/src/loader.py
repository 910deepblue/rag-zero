# -*- coding: utf-8 -*-
"""
doc_loader.py —— 文档加载模块

职责：把不同格式的本地文档(PDF / 纯文本 / Markdown / Word)统一读成
      「文本单元(unit)」列表，交给下游做 切块(chunk) → 向量化 → 入库。

统一的数据结构(所有 loader 的返回值都长这样)：
    [{"text": "正文...", "source": "文件.pdf", "page": 3}, ...]
    - text   : str，该单元正文(已 strip；空内容会被丢弃)
    - source : str，来源文件名(只取 name、不带目录，方便在答案里标注出处)
    - page   : int，页码，从 1 开始；对没有"页"概念的格式(txt/md/docx)固定填 0
"""

from pathlib import Path

# ⚠️ 【修订 5】新版 PyMuPDF 官方推荐 import pymupdf。
#    旧写法 `import fitz` 仍然可用，但会打弃用警告：
#      warning: The `fitz` API is deprecated and will be removed in future.
#               Use `import pymupdf` instead.
import pymupdf


def load_pdf(path: Path) -> list[dict]:
    """逐页抽取文字，返回 [{text, source, page}]"""
    out = []                                    # 累积结果
    # with 管理句柄：正常/异常退出都会 close，避免文件被占用或句柄泄漏
    with pymupdf.open(path) as doc:
        # enumerate(doc, start=1)：可直接迭代文档拿到每页对象，页码从 1 开始(非 0)
        for pno, page in enumerate(doc, start=1):
            # get_text("text") 取纯文本流；换成 "blocks"/"dict" 可拿到带坐标/字体的结构
            text = page.get_text("text").strip()
            # 空白页、纯图片扫描页(没有文字层)会得到空串 → 直接跳过，不给下游留噪声
            if text:
                out.append({"text": text, "source": path.name, "page": pno})
    return out


def load_text(path: Path) -> list[dict]:
    """读取 txt / md：整个文件作为一个单元(这里不切分，切分交给下游)"""
    # errors="ignore"：遇到非 UTF-8 的字节不抛异常、直接丢弃，保证整批流程不中断
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    # 空文件返回 []（而不是 [{"text": ""}]），下游就不用再判空
    return [{"text": text, "source": path.name, "page": 0}] if text else []


def load_docx(path: Path) -> list[dict]:
    """读取 Word(.docx)：段落 + 表格文字都要，整篇作为一个单元"""
    from docx import Document                  # 局部导入：只有真读 docx 时才依赖 python-docx
    doc = Document(str(path))                  # python-docx 只接受 str，不吃 Path 对象
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # 表格里的文字也别漏：docx 中 tables 和 paragraphs 是两套独立结构，
    # 只遍历 paragraphs 会整片丢掉表格内容
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))    # 用 " | " 拼成一行，保留列的关系

    text = "\n".join(parts)
    return [{"text": text, "source": path.name, "page": 0}] if text else []


# 后缀 → 处理函数的映射表。
# 新增格式(如 .csv/.html)只要在这里加一行，load_dir 完全不用改(对扩展开放)
LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".docx": load_docx,
}


def load_dir(data_dir: Path) -> list[dict]:
    """遍历目录，返回所有文档单元（每个单元含来源信息）"""
    units = []
    # rglob("*") 递归遍历所有子层级；sorted 保证每次运行顺序一致(可复现，便于调试比对)
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():                 # 跳过目录本身(rglob 也会命中目录)
            continue

        loader = LOADERS.get(path.suffix.lower())   # lower() 兼容 .PDF / .Docx 这类大写后缀
        if loader is None:
            print(f"[跳过] 不支持的类型: {path.name}")
            continue

        # 单个文件出错不影响整体：读失败只打日志，继续处理下一个文件
        try:
            got = loader(path)
            print(f"[读取] {path.name} -> {len(got)} 个单元")
            units.extend(got)                  # extend 把该文件的所有单元并入总列表
        except Exception as e:
            print(f"[失败] {path.name}: {e}")
    return units
