from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"
INDEX_PATH = STORAGE_DIR / "index.faiss"
CHUNK_META_PATH = STORAGE_DIR / "chunks.jsonl"

# 向量模型：本地路径优先，找不到再退回 HuggingFace 名称
_LOCAL_MODEL = Path(r"D:\DevEnv\ai-cache\models\bge-small-zh-v1.5")
EMBEDDING_MODEL = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "BAAI/bge-small-zh-v1.5"

# BGE 系列检索时建议给 query 加指令前缀，能明显提升召回
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

CHUNK_SIZE = 800      # 每块目标字符数
CHUNK_OVERLAP = 120    # 相邻块重叠字符数
TOP_K = 5             # 检索取前几块