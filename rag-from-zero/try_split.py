from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_dir
from src.splitter import split_units

units = load_dir(DATA_DIR)
chunks = split_units(units, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"\n文档单元 {len(units)} 个 -> 切分成 {len(chunks)} 块")

for c in chunks[:8]:
    print("-" * 60)
    print(f"[{c['source']} p{c['page']} part{c['part']}] 长度 {len(c['text'])}")
    print(c["text"][:200])