import sys
print("Python:", sys.version.split()[0])

import numpy, torch, faiss, fitz, sentence_transformers
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("faiss:", getattr(faiss, "__version__", "unknown"))
print("pymupdf:", fitz.__doc__)
print("sentence-transformers:", sentence_transformers.__version__)

from pathlib import Path
p = Path(r"D:\DevEnv\ai-cache\models\bge-small-zh-v1.5")
print("本地模型存在:", p.exists(), "文件数:", len(list(p.iterdir())) if p.exists() else 0)