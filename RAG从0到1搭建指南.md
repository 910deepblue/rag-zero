# RAG 从 0 到 1 搭建指南

> 面向对象：会一点 Python、想亲手做一个「能回答我自己资料」的问答系统的人。
> 目标：跟着做完，你手上会有一个能跑、能换资料、能持续优化的 RAG 项目。
> 环境：Windows + 本机已有资源（Anaconda / uv / 本地 BGE 模型）。

---

## ⚠️ 阅读前必看：本机现状核对表 + 修订记录

**这份文档写于环境搭建完成之前。环境搭好后实测发现若干处与文档不符，已逐条修正并打上标记。
凡带 `⚠️ 【修订 N】` 或 `🔴` 的段落，都是改过或需要特别注意的地方，务必按标记后的内容执行。**

### 本机现状（2026-09-14 实测确认，与文档原文有出入）

| 项 | 实际值 | 与原文的差异 |
| --- | --- | --- |
| 项目位置 | **`E:\rag-from-zero`** | 原文写的是工作区目录，该目录已不存在 |
| 虚拟环境 | `E:\rag-from-zero\.venv`，**Python 3.12.7** | 原文写 3.11 |
| torch | **2.6.0+cpu**（CPU 版，`cuda.is_available()` = `False`） | 由 `constraints.txt` 锁死 |
| 向量模型 | `D:\DevEnv\ai-cache\models\bge-small-zh-v1.5`，**输出 512 维** | 原文写 384 维（错） |
| sentence-transformers | **6.0.1**（API 名有变） | 原文用的是旧 API 名 |
| 环境体积 | `.venv` 约 1.6GB（其中 `torch/` 占 1.1GB） | — |
| 磁盘余量 | C: ≈19GB / E: ≈58GB / F: ≈46GB | C 盘紧张，**缓存别往 C 盘放** |

### 修订记录（共 8 处 + 2 处新增坑）

| # | 位置 | 改了什么 | 为什么（坑） | 级别 |
| --- | --- | --- | --- | --- |
| 1 | §2.1 / §2.2 | 路径改为 `E:\rag-from-zero`；Python `3.11` → **`3.12`** | 原路径已不存在；实测建出来的是 3.12.7 | 🟠 |
| 2 | §2.2 | **新增** uv 缓存/下载目录环境变量设置 | 不设的话 wheel 全下到只剩 19GB 的 C 盘 | 🟠 |
| 3 | **§2.3** | 安装命令从「两条命令 + 两个源」改为**「一条命令 + 锁版本」** | 🔴 **原写法会装出"缝合怪" torch，直接 `ImportError` 起不来** | 🔴 |
| 4 | §2.3 | **新增** `constraints.txt` 文件及用法 | 锁死 torch 版本，是防"缝合怪"的关键 | 🔴 |
| 5 | §2.5 / §4.2 | `import fitz` → `import pymupdf` | 新版 PyMuPDF 弃用了 `fitz` 这个名字，会打警告 | 🟡 |
| 6 | **§3** | 向量维度 **384 → 512** | 原文 384 是错的，实测 512 | 🔴 |
| 7 | §5.1 | `get_sentence_embedding_dimension()` → **`get_embedding_dimension()`** | sentence-transformers 6.x 改了 API 名 | 🟠 |
| 8 | §5.1 | **补充** `device="cuda"` 的前提说明 | 装的是 CPU 版 torch，直接改 `cuda` 会报错 | 🟠 |
| — | §2.5 | 补上本机实测的**正确输出样例** | 便于对照判断环境是否真的 OK | 🟡 |
| — | **第 9 章** | **新增坑 0**：torch 缝合怪 · **坑 0.1**：torch 下载慢 | 本机遇到概率最高的两个坑 | 🔴 |
| — | 附录 | 文件清单补上 `constraints.txt` | 缺了会漏掉关键约束文件 | 🟡 |

---

## 目录

- [第 0 章 先搞清楚 RAG 是什么](#第-0-章-先搞清楚-rag-是什么)
- [第 1 章 技术选型（为什么这么选）](#第-1-章-技术选型为什么这么选)
- [第 2 章 建项目骨架与环境](#第-2-章-建项目骨架与环境)
- [第 3 章 里程碑 1：30 行跑通最小 RAG](#第-3-章-里程碑-130-行跑通最小-rag)
- [第 4 章 里程碑 2：把你的文档灌进去](#第-4-章-里程碑-2把你的文档灌进去)
- [第 5 章 里程碑 3：向量化 + 落盘（真正的索引）](#第-5-章-里程碑-3向量化--落盘真正的索引)
- [第 6 章 里程碑 4：接上大模型，输出带引用的答案](#第-6-章-里程碑-4接上大模型输出带引用的答案)
- [第 7 章 里程碑 5：让效果变好（检索优化）](#第-7-章-里程碑-5让效果变好检索优化)
- [第 8 章 里程碑 6：工程化与评估](#第-8-章-里程碑-6工程化与评估)
- [第 9 章 本机高频坑清单](#第-9-章-本机高频坑清单)
- [第 10 章 学习路线](#第-10-章-学习路线)

---

## 第 0 章 先搞清楚 RAG 是什么

一句话：**RAG = 检索（Retrieval）+ 增 强（Augmented）+ 生成（Generation）**。

大模型本身不知道你的私人资料，而且会一本正经地编。RAG 的做法是：**别让模型凭空回答，先在你的资料库里搜出相关段落，把段落塞进提示词，再让模型"照着资料"回答。**

它和另外两条路线的区别：

| 方案 | 做法 | 适合 | 代价 |
| --- | --- | --- | --- |
| 提示词硬塞 | 把资料直接贴进 prompt | 资料只有几页 | 上下文放不下 |
| 微调（Fine-tune） | 用资料训练模型 | 要改变说话风格 / 固定任务 | 贵、更新资料要重训 |
| **RAG** | 检索后拼进 prompt | **资料经常变、要引用出处** | 需要一套检索工程 |

为什么绝大多数「企业知识库问答」选 RAG：**资料改了只要重建索引，几秒钟的事，模型一个字都不用重训。**

### 两条链路

整个系统其实只有两条链路，务必在脑子里分开：

- **离线链路（建索引，跑一次或资料更新时跑）**：文档 → 解析 → 切分 → 向量化 → 存进向量库
- **在线链路（问答，用户每问一次跑一次）**：问题 → 向量化 → 检索 Top-K → 拼提示词 → 大模型生成

新手最容易犯的错：把两条链路混在一个脚本里，每次提问都重新解析 PDF 重新编码。**这两条链路必须拆开。**

---

## 第 1 章 技术选型（为什么这么选）

先给结论表，再解释。

| 环节 | 本指南选型 | 理由 | 以后可以换 |
| --- | --- | --- | --- |
| 文档解析 | PyMuPDF (`fitz`) | 中文 PDF 抽取质量最好，速度极快，单库搞定 | MinerU、RapidOCR（扫描件） |
| 文本切分 | **手写** 递归切分 | 30 行代码，理解切分策略是 RAG 的核心功 | LangChain `RecursiveCharacterTextSplitter` |
| 向量模型 | **BGE-small-zh-v1.5**（本机已有） | 中文效果好、只有 95MB、CPU 跑几十毫秒 | BGE-large-zh、m3e、通义 embedding API |
| 向量库 | **FAISS** (IndexFlatIP) | 单文件、零部署、十万级向量毫秒级检索 | Chroma、Qdrant、Milvus |
| 生成模型 | OpenAI 兼容 API（通义千问）/ 本地 Ollama | 一句话就能换模型，不用改代码 | 任何 OpenAI 兼容服务 |
| 编排框架 | **不用**，先手写 | 手写过一遍，才知道框架在帮你做什么 | LlamaIndex、LangChain（第 8 章再上） |

### 三个关键决策解释

**1. 为什么先手写、不上 LangChain / LlamaIndex？**

框架把「加载、切分、嵌入、检索、生成」全封装成 `chain`，你三行就能跑出结果，于是你以为自己会 RAG 了。但一旦效果不好，你不知道该调哪里。**先用 300 行手写一遍完整链路，再上框架，你会用得比 90% 的人好。**

**2. 为什么向量库先选 FAISS 而不是 Milvus？**

FAISS 是库不是服务，`pip install` 就能用，索引就是一个文件。Milvus / Qdrant 是服务，要 Docker、要端口、要运维。你的资料量在一万条以内，FAISS 完全够，而且快。

**3. 为什么 embedding 用本地模型，生成用 API？**

- embedding 是高频、小算力任务（一次问答要算 1 次，建库要算 N 次），本地跑免费、无网络延迟、不受限流。BGE-small 在 CPU 上跑一句话约 10~30ms。
- 生成是低频、大算力任务，本地跑 7B 模型在你 2060（6G 显存）上要挤显存、还会慢。用 API 又快又好，还便宜。
- **这个组合是性价比最高的起步方案。**

---

## 第 2 章 建项目骨架与环境

### 2.1 目录结构（先建好，别乱放）

```text
rag-from-zero/
├─ data/                    # 放原始资料（pdf / txt / md / docx）
├─ storage/                 # 索引产物（自动生成）
│   ├─ index.faiss          # 向量索引
│   └─ chunks.jsonl         # 每个向量的元数据（原文、来源、页码）
├─ src/
│   ├─ config.py            # 所有配置集中在这
│   ├─ loader.py            # 文件 → 纯文本
│   ├─ splitter.py          # 纯文本 → 块
│   ├─ embedder.py          # 块 → 向量
│   ├─ vector_store.py      # 向量的存与查
│   ├─ retriever.py         # 检索（后续加混合检索、重排）
│   └─ llm.py               # 调大模型
├─ build_index.py           # 离线链路入口
├─ ask.py                   # 在线链路入口
├─ check_env.py             # 环境自检
├─ requirements.txt
├─ constraints.txt          # ⚠️ 锁定 torch 版本，防止装出"缝合怪"（见 2.3）
└─ .env                     # 密钥（不要提交到 git）
```

> ⚠️ **【修订 1】项目位置改了：`E:\rag-from-zero`**
>
> 原文写的是工作区目录 `E:\腾讯agent\2026-09-14-14-32-25\rag-from-zero`，那个目录已经不存在了。
> **本文档后续所有命令统一按 `E:\rag-from-zero` 写。** 这一步你已经做完了，下面的命令只是留档备查。

在 PowerShell 里建：

```powershell
# ⚠️ 【修订 1】实际路径是 E:\rag-from-zero（不是工作区里那个）
mkdir E:\rag-from-zero
cd E:\rag-from-zero
mkdir data, storage, src
ni src\__init__.py -ItemType File
```

> ⚠️ **【修订 1】注意**：目录名和位置**随你放，但一旦定了，后面所有命令的 `cd` 都要跟着改**。
> 凡是文档里出现 `E:\rag-from-zero` 的地方，如果你换了位置，都要替换成你自己的路径。

### 2.2 虚拟环境

你有 uv，用 uv 最快（它自动管理 Python 版本、装包比 pip 快很多）：

```powershell
# ⚠️ 【修订 1】路径改为 E:\rag-from-zero；Python 版本 3.11 → 3.12
cd E:\rag-from-zero

# ⚠️ 【修订 2】先把 uv 的缓存 / 下载 / 工具目录全指到 D 盘
#            否则 torch 那个 200MB 的 wheel 会下到只剩 19GB 的 C 盘
$env:UV_CACHE_DIR          = "D:\DevEnv\uv\cache"
$env:UV_PYTHON_INSTALL_DIR = "D:\DevEnv\uv\python"
$env:UV_TOOL_DIR           = "D:\DevEnv\uv\tools"

D:\DevEnv\uv\bin\uv.exe venv --python 3.12
.\.venv\Scripts\Activate.ps1
```

> ⚠️ **【修订 2】注意：这三行 `$env:` 只对当前 PowerShell 窗口生效**，关掉窗口就没了。
> 每开一个新窗口准备装包，都得重新设一遍（想永久生效就写进系统"用户环境变量"）。
> **不设的后果**：uv 把缓存下到 `C:\Users\<你的用户名>\AppData\Local\uv\cache`，而你 C 盘只剩 19GB。

> ⚠️ **【修订 1】注意：本机实测建出来的是 Python 3.12.7。**
> 原文写的 3.11 也能用，但 3.12 的 wheel 在 uv 缓存里**现成已经有**（复用现成的 CPython，不额外下载），所以改成了 3.12。

如果 PowerShell 提示「禁止运行脚本」，就在当前窗口执行一次：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

（临时生效，关掉窗口即失效，安全。）激活成功后，命令行前面会出现 `(rag-from-zero)`。

> 不想用 uv 就退回 venv（把 `python` 换成你 Anaconda 里的解释器也行）：
> ```powershell
> D:\Anaconda\envs\mindspore\python.exe -m venv .venv
> ```

### 2.3 依赖清单

写 `requirements.txt`：

```text
# --- 核心 ---
numpy
sentence-transformers>=3.0
faiss-cpu
pymupdf
openai
python-dotenv
tqdm
# --- 文档格式 ---
python-docx
# --- 第 7 章检索优化才需要 ---
jieba
rank_bm25
# --- 第 8 章做界面才需要 ---
# streamlit
```

### 安装依赖

> 🔴 **【修订 3】这一节原文是错的，照原样做会直接把环境装坏。**
>
> **原文的写法（已废弃，别用）**：
> ```powershell
> uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # 先装 CPU 版 torch
> uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```
>
> **为什么坏**：第二条命令**换了索引源**，而 `requirements.txt` 里的 `sentence-transformers` 依赖 torch
> —— uv 会在清华源里**再解析出一个 CUDA 版 torch**，然后拿去覆盖第一次装的 CPU 版。
> **覆盖不干净**，新旧两批文件混在同一个 `torch/` 目录里，就成了「缝合怪」。症状是：
> ```text
> ImportError: cannot import name 'thread_safe_generator' from 'torch.random'
> ```
> 更气人的是：`uv pip install` 会打印 `Checked 10 packages in 449ms`，**让你以为装成功了**，实际 `import torch` 当场崩。
>
> ⚠️ **一句话教训：同一个包，绝对不要用两个不同的索引源分两次装。**

---

⚠️ **【修订 4】正确做法：先写 `constraints.txt` 锁死 torch，再用一条命令装完。**

**第一步**：在 `E:\rag-from-zero` 下新建 `constraints.txt`：

```text
# 锁死 torch 为 CPU 版：无论 uv 走哪个索引源，都不许动 torch 的版本
torch==2.6.0+cpu
```

**第二步**：**一条命令**装完所有依赖：

```powershell
cd E:\rag-from-zero
D:\DevEnv\uv\bin\uv.exe pip install -r requirements.txt -c constraints.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple `
  --extra-index-url https://download.pytorch.org/whl/cpu `
  --index-strategy unsafe-best-match `
  -p "E:\rag-from-zero\.venv\Scripts\python.exe"
```

**四个参数缺一不可，各自的用途**：

| 参数 | 作用 | 删掉会怎样 |
| --- | --- | --- |
| `-c constraints.txt` | **锁版本**。torch 只可能被解析成一次 `2.6.0+cpu`，从根上杜绝缝合怪 | 退化成原文的坏写法 |
| `--extra-index-url .../whl/cpu` | 多给一个源，让 CPU 版 torch 能被找到 | 清华源上没有带 `+cpu` 后缀的包，找不到 torch |
| `--index-strategy unsafe-best-match` | 允许多源混查时取"最优匹配" | uv 会因源冲突直接报错退出 |
| `-p <venv 里的 python>` | 明确指定装进哪个环境 | 可能装到你系统 Python 里去 |

> ⚠️ **【修订 3】注意 A：这条命令很慢，必须后台跑，而且看不到进度条。**
> - `torch 2.6.0+cpu` 的 wheel 约 **197MB**，只在 `download.pytorch.org`（**境外**）上；
> - 国内直连实测只有 **0.1~0.5 MB/s**，**整条命令实测跑完 72 分钟**（其中约 70 分钟都在下 torch）；
> - **前台跑会被系统掐断**，表现是"命令没输出就结束了"，让你以为失败；
> - 后台跑起来后**看不到进度条**（uv 的输出被重定向时就不画进度条了），**别以为它卡死了**——
>   实测用进程 I/O 计数能看到数据在持续流入。
>
> ⚠️ **【修订 3】注意 B：开个代理会快几十倍。**
> 实测到该源的速度：未开代理 ~0.1~0.5 MB/s → 开代理（TUN 模式，全局生效）后 **5.9~18.7 MB/s**，
> torch 两三分钟就下完。**如果还有下次重建，先把代理开上再跑。**
>
> ✅ **本机实测的完成标志**（照这个对照）：
> ```text
> Resolved 59 packages in 17.23s
> Installed 16 packages in 72m 08s
>  + torch==2.6.0+cpu
>  ...
> === EXIT: 0 ===
> ```
> **`Installed` 列表里出现 `torch==2.6.0+cpu`，且没有第二个 torch 条目**，才算干净。

### 2.4 关键一步：让模型走本地，绕开下载

你已经把模型放在 `D:\DevEnv\ai-cache\models\bge-small-zh-v1.5`，**代码里直接传这个路径**，`SentenceTransformer` 会把它当本地目录加载，完全不联网。这是最稳的做法。

如果以后要下新模型，国内用镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

写进 `.env` 也可以（但注意第 9 章的坑）。

### 2.5 环境自检

`check_env.py`：

```python
import sys
print("Python:", sys.version.split()[0])

# ⚠️ 【修订 5】改用 import pymupdf（旧写法 import fitz 仍能用，但会打弃用警告）
import numpy, torch, faiss, pymupdf, sentence_transformers
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("faiss:", getattr(faiss, "__version__", "unknown"))
print("pymupdf:", pymupdf.__doc__)
print("sentence-transformers:", sentence_transformers.__version__)

from pathlib import Path
p = Path(r"D:\DevEnv\ai-cache\models\bge-small-zh-v1.5")
print("本地模型存在:", p.exists(), "文件数:", len(list(p.iterdir())) if p.exists() else 0)
```

跑：

```powershell
cd E:\rag-from-zero
.\.venv\Scripts\python.exe check_env.py
```

> ℹ️ **【新增】本机实测的"正确输出"长这样，照这个比对：**
> ```text
> Python: 3.12.7
> numpy: 2.5.3
> torch: 2.6.0+cpu cuda: False
> faiss: 1.15.0
> pymupdf: PyMuPDF 1.28.2: Python bindings for the MuPDF 1.28.2 library.
> Python 3.12 running on win32 (64-bit).
> sentence-transformers: 6.0.1
> 本地模型存在: True 文件数: 10
> ```

全部打印出来、最后一行是 `True`，就可以进第 3 章了。**任何一行报 ImportError，先回去解决，不要往前走。**

> ⚠️ **【新增】两条注意**：
> 1. **`cuda: False` 是正常的**，本项目装的就是 CPU 版 torch（见第 1 章选型、第 2.3 节安装）。**不要试图去"修"它**，改成 CUDA 版要额外下 2.4GB，且对本项目没用（理由见 §5.1 修订 8）。
> 2. **`uv pip install` 输出 `Checked N packages` 不代表包可用**（见 §2.3 的缝合怪坑）。
>    **只有 `import` 成功才算数** —— 这正是必须有这个自检脚本的原因。

---

## 第 3 章 里程碑 1：30 行跑通最小 RAG

**这一步的目标不是做产品，是让你亲眼看到"向量检索"到底在干什么。** 不落盘、不接大模型、不读文件，纯内存。

新建 `mini_rag.py`：

```python
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
```

跑：

```powershell
cd E:\rag-from-zero
.\.venv\Scripts\python.exe mini_rag.py
```

期望看到 `FAISS...` 那一行排第一，分数大概 0.6~0.8。

> ⚠️ **【新增】注意：第一次 `import` 要等约 15 秒，别以为卡死了。**
> 本机实测 `from sentence_transformers import SentenceTransformer` **冷启动 14.8 秒**（时间都花在导入 torch 上）。
> 这是 CPU 版 torch 的正常现象，**不是故障**。模型真正加载只花 **0.4 秒**，编码 5 条文档 **0.05 秒**。

### 你必须从这一步带走的三个认知

1. **"检索"根本不是魔法**，就是向量点积排序。`doc_vecs @ q_vec` 这一行就是 RAG 的心脏。
2. **向量维度**：⚠️ **【修订 6】BGE-small-zh-v1.5 输出的是 512 维**（原文写的"384 维"是**错的** —— 384 是 `bge-small-en` / MiniLM 那一类**英文**小模型的维度）。
   自己加一句 `print(doc_vecs.shape)` 确认，**本机实测是 `(5, 512)`**。别被文档写错的数字带偏。
3. **它只做"找相似"，完全不懂"对不对"**。如果你问「中国的首都有多少人」，它可能给回北京那句（对），也可能给回 FAISS 那句（因为都含"检索/搜索"）。**RAG 效果不好的根源，八成在检索，不在模型。**

> 常见问题：`OSError: Can't load tokenizer`。
> 原因通常是路径写错，或者传了 HuggingFace 的模型名导致联网下载失败。**检查那个路径下有没有 `config.json`、`vocab.txt`、`model.safetensors`。**

---

## 第 4 章 里程碑 2：把你的文档灌进去

现在把"硬编码的 docs 列表"换成"真实文件"。这一步做两件事：**解析** 和 **切分**。

### 4.1 配置集中管理

`src/config.py`：

```python
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

CHUNK_SIZE = 500      # 每块目标字符数
CHUNK_OVERLAP = 80    # 相邻块重叠字符数
TOP_K = 5             # 检索取前几块
```

> 把配置集中在一个文件，是新手项目最容易忽略、后期最救命的一件事。等你开始调 chunk_size 的时候会感谢自己。

### 4.2 文档解析

`src/loader.py`：

```python
from pathlib import Path
# ⚠️ 【修订 5】新版 PyMuPDF 官方推荐 import pymupdf。
#    旧写法 `import fitz` 仍然可用，但会打弃用警告：
#      warning: The `fitz` API is deprecated and will be removed in future.
#               Use `import pymupdf` instead.
import pymupdf


def load_pdf(path: Path) -> list[dict]:
    """逐页抽取文字，返回 [{text, source, page}]"""
    out = []
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                out.append({"text": text, "source": path.name, "page": pno})
    return out


def load_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"text": text, "source": path.name, "page": 0}] if text else []


def load_docx(path: Path) -> list[dict]:
    from docx import Document
    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    # 表格里的文字也别漏
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n".join(parts)
    return [{"text": text, "source": path.name, "page": 0}] if text else []


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".docx": load_docx,
}


def load_dir(data_dir: Path) -> list[dict]:
    """遍历目录，返回所有文档单元（每个单元含来源信息）"""
    units = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"[跳过] 不支持的类型: {path.name}")
            continue
        try:
            got = loader(path)
            print(f"[读取] {path.name} -> {len(got)} 个单元")
            units.extend(got)
        except Exception as e:
            print(f"[失败] {path.name}: {e}")
    return units
```

**要点**：解析阶段一定要保留 `source` 和 `page`。这是后面做**引用溯源**的唯一依据，丢了就补不回来了。

### 4.3 切分

`src/splitter.py`：

```python
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
```

**怎么定 chunk_size？**
- 太**小**（<200 字）：召回的是碎片，模型看不到完整逻辑，答一半。
- 太**大**（>1000 字）：一块里混了好几个主题，检索精度掉，还浪费 token。
- **中文起步值：400~600 字，overlap 取 chunk_size 的 10%~20%。**
- 这不是玄学，是**要调的参数**。第 8 章会教你怎么客观比较。

### 4.4 先验证解析 + 切分

在建索引之前，先单独看结果（`try_split.py`）：

```python
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_dir
from src.splitter import split_units

units = load_dir(DATA_DIR)
chunks = split_units(units, CHUNK_SIZE, CHUNK_OVERLAP)
print(f"\n文档单元 {len(units)} 个 -> 切分成 {len(chunks)} 块")

for c in chunks[:3]:
    print("-" * 60)
    print(f"[{c['source']} p{c['page']} part{c['part']}] 长度 {len(c['text'])}")
    print(c["text"][:200])
```

把一两个 PDF / txt 丢进 `data/` 目录，然后 `python try_split.py`。

**眼睛看一遍切出来的块**：有没有把一句完整的话切成两半？有没有一块里塞了三个小标题？这一步花五分钟，能省你后面两小时排查。

---

## 第 5 章 里程碑 3：向量化 + 落盘（真正的索引）

### 5.1 向量化封装

`src/embedder.py`：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, QUERY_INSTRUCTION


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        # ⚠️ 【修订 7】sentence-transformers 6.x 把旧名 get_sentence_embedding_dimension()
        #    改成了 get_embedding_dimension()。旧名还能用，但每次都会打 FutureWarning：
        #      FutureWarning: The `get_sentence_embedding_dimension` method has been renamed
        #                     to `get_embedding_dimension`.
        #    下面这样写，新旧版本都能跑（本机装的是 6.0.1，走新名字这条路）。
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # 归一化后内积 == 余弦相似度
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        # 检索侧加指令前缀，BGE 官方推荐做法，召回更准
        vec = self.model.encode(
            [QUERY_INSTRUCTION + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype("float32")
```

**关于 `device`**：默认 `"cpu"`。BGE-small 在 CPU 上处理一万块也就一两分钟。你有 2060，想用 GPU 就把 device 改成 `"cuda"`——但注意你的 mindspore 环境可能占着显存，而且小模型用 GPU 提升有限，`batch_size` 也调小点（16）更稳。

> 🟠 **【修订 8】注意：想改 `device="cuda"`，必须先把 torch 换成 CUDA 版，否则直接报错。**
>
> 本项目按 §2.3 装的是 **CPU 版 torch**（`torch.cuda.is_available()` 是 `False`）。
> **在 CPU 版 torch 上写 `device="cuda"` 不是"慢一点"，是当场崩。** 原文没写这个前提，容易误导。
>
> 真要上 GPU 的两步（**必须在同一个环境里整体替换，绝不能两个源混装**——那就是 §2.3 缝合怪的成因）：
> ```powershell
> # 1) 把 constraints.txt 里的 torch==2.6.0+cpu 改成 torch==2.6.0
> # 2) 用 CUDA 源整体重装（RTX 2060 是 sm_75，cu124 支持）
> D:\DevEnv\uv\bin\uv.exe pip install torch==2.6.0 `
>   --index-url https://download.pytorch.org/whl/cu124 `
>   -p "E:\rag-from-zero\.venv\Scripts\python.exe"
> ```
> **装完先验证再改代码**：
> ```powershell
> .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
> ```
> 输出 `True` 加一个版本号，才可以动 `device`。
>
> ⚠️ **但对本项目并不划算**：本地只跑一个 **95MB 的小 embedding 模型**，LLM 走 API，GPU 帮不上大忙，
> 却要多下 **2.4GB** 的 wheel、多占 **3~4GB** 磁盘。
> **你系统里装的 CUDA 12.9.2 与 torch 无关** —— PyTorch 的 wheel 自带 CUDA 运行时，不依赖系统那套。

### 5.2 向量库

`src/vector_store.py`：

```python
import json
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    """极简本地向量库：FAISS 存向量，jsonl 存原文与来源（行号即 id）"""

    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index: faiss.Index | None = None
        self.metas: list[dict] = []

    # ---------- 写入 ----------
    def build(self, vectors: np.ndarray, metas: list[dict]) -> None:
        assert vectors.shape[0] == len(metas), "向量数与元数据数不一致"
        dim = vectors.shape[1]
        # 向量已归一化 -> 用内积索引，等价于余弦相似度
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metas = metas
        self.save()
        print(f"[建库] {len(metas)} 条, 维度 {dim} -> {self.index_path}")

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- 读取 ----------
    def load(self) -> "FaissStore":
        self.index = faiss.read_index(str(self.index_path))
        self.metas = [
            json.loads(line)
            for line in self.meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return self

    # ---------- 查询 ----------
    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        if self.index is None:
            self.load()
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, idx = self.index.search(q, k)
        hits = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            hits.append((float(score), self.metas[i]))
        return hits
```

**为什么元数据用 jsonl 而不是数据库？**

- 行号天然是 id，和 FAISS 的整数 id 一一对应，不会错位；
- 追加方便（以后做增量索引就靠它）；
- 用记事本就能打开检查，调试友好。
- 上万条以内完全够用；到了几十万条再换 SQLite / PostgreSQL。

### 5.3 离线链路入口

`build_index.py`：

```python
from src.config import (DATA_DIR, INDEX_PATH, CHUNK_META_PATH,
                        CHUNK_SIZE, CHUNK_OVERLAP)
from src.loader import load_dir
from src.splitter import split_units
from src.embedder import Embedder
from src.vector_store import FaissStore


def main():
    print("=" * 50, "\n[1/4] 解析文档")
    units = load_dir(DATA_DIR)
    if not units:
        print("data/ 目录里没有可用文档，先放几个进去。")
        return

    print("=" * 50, "\n[2/4] 切分")
    chunks = split_units(units, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"共 {len(chunks)} 块")

    print("=" * 50, "\n[3/4] 向量化")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode_docs(texts)

    print("=" * 50, "\n[4/4] 建库落盘")
    FaissStore(INDEX_PATH, CHUNK_META_PATH).build(vectors, chunks)
    print("完成。")


if __name__ == "__main__":
    main()
```

跑：

```powershell
python build_index.py
```

成功后 `storage/` 下应该有 `index.faiss` 和 `chunks.jsonl`。**打开 `chunks.jsonl` 看一眼**——这就是你的知识库实体，所有检索都基于它。

> 索引重建：`IndexFlatIP` 是暴力检索索引，**不支持删除单条**。所以现阶段策略是「资料变了就整个重建」。一万块以内重建一两分钟，够用。真要做增量，在第 8 章。

---

## 第 6 章 里程碑 4：接上大模型，输出带引用的答案

### 6.1 配置密钥

`.env`：

```text
# 方案 A：通义千问（DashScope，OpenAI 兼容）
DASHSCOPE_API_KEY=sk-你的key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# 方案 B：本地 Ollama（先 ollama pull qwen2.5:7b-instruct）
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=qwen2.5:7b-instruct
# DASHSCOPE_API_KEY=ollama
```

`src/llm.py`：

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
MODEL = os.getenv("LLM_MODEL", "qwen-plus")


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,   # 问答要稳，温度调低
    )
    return resp.choices[0].message.content
```

**注意 `base_url` 这一行**：只要服务兼容 OpenAI 协议（通义、DeepSeek、Kimi、硅基流动、Ollama 都兼容），换模型只是改这两行，代码一个字不用动。这是选 OpenAI SDK 而不是各家 SDK 的核心原因。

### 6.2 提问脚本

`ask.py`：

```python
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
```

跑：

```powershell
python ask.py 你文档里的某个问题
```

**到这里，一个功能完整的 RAG 就跑通了。** 剩下的全是优化。

### 6.3 三个立刻能感觉到的效果差异

把 `TOP_K` 从 5 改成 1，再改成 10，问同一个问题，观察：
- K 太小 → 该给的资料没给，模型说"未提及"；
- K 太大 → 塞进一堆无关内容，模型被带偏，还费 token；
- **K=5 通常是个不错的起点，但要按你的资料密度调。**

---

## 第 7 章 里程碑 5：让效果变好（检索优化）

按**性价比从高到低**排序，别跳步。

### 7.1 优化一：Prompt 约束与引用（成本 0，收益巨大）

很多"RAG 幻觉"不是检索问题，是 prompt 没约束。检查三件事：
1. 有没有明确写出"资料中没有就说未提及"？
2. 有没有要求模型标注引用编号？
3. 有没有把 `temperature` 调到 0.1~0.3？

再加一条实战经验：**把资料放在问题前面、并在资料外包裹明确标记**（就像上面模板那样），比反过来效果好。

### 7.2 优化二：混合检索（关键词 + 向量）

**纯向量检索的软肋**：专有名词、型号编号、人名、代码标识符。比如你问"XT-9000 的保修期"，向量模型可能觉得这句话和"设备保修政策"很像，于是召回错的东西；而关键词检索能精准命中"XT-9000"。

用 RRF（倒数排名融合）把两路结果合并，不需要调权重，非常省心：

```powershell
uv pip install jieba rank_bm25 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`src/retriever.py`：

```python
import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class HybridRetriever:
    """向量检索 + BM25 关键词检索，用 RRF 融合两路排名"""

    def __init__(self, embedder, store, chunks: list[dict]):
        self.embedder = embedder
        self.store = store
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[tuple[float, dict]]:
        # --- 路 1：向量 ---
        vec_hits = self.store.search(self.embedder.encode_query(query), k=k * 4)
        vec_docs = [m["id"] for _, m in vec_hits]

        # --- 路 2：BM25 ---
        scores = self.bm25.get_scores(_tokenize(query))
        bm25_docs = sorted(range(len(scores)), key=lambda i: -scores[i])[: k * 4]

        # --- RRF 融合：只看排名，不看绝对分数 ---
        fused: dict[int, float] = {}
        for rank, doc_id in enumerate(vec_docs):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(bm25_docs):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(score, self.chunks[doc_id]) for doc_id, score in top]
```

**RRF 为什么好用**：向量的余弦相似度是 0.6~0.9 的连续值，BM25 是 0~30 的分数，两者**量纲完全不同，没法直接相加**。RRF 只取"排第几名"，`1/(60+rank)`，天然免疫量纲问题，还几乎不需要调参。

### 7.3 优化三：重排（Rerank）

**召回（retrieval）和排序（ranking）是两件事。** 向量模型为了快，用的是"双塔结构"——问题和文档分别编码，从不交互，所以精度有限。重排模型用"交叉编码"——把问题和文档拼在一起过一遍模型，**精度高得多，但慢**。

所以标准做法是：**召回阶段用向量/BM25 捞 20~50 条（粗筛），重排阶段用 cross-encoder 精排出前 5 条。** 这就是"两阶段检索"。

```python
# src/reranker.py
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, cands: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, c["text"]) for c in cands]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[:top_k]
        return [c for _, c in ranked]
```

首次使用会从 HuggingFace 下载，国内记得先设 `$env:HF_ENDPOINT = "https://hf-mirror.com"`。**注意 bge-reranker-base 约 1.1GB，比 embedding 模型大得多**；单条推理在 CPU 上约 50~200ms，检索 20 条就是 1~4 秒，可以接受。

不想下大模型，可以用 API 版重排（通义 `gte-rerank`），或者先用第 7.2 节的混合检索 —— **对绝大多数自用场景，混合检索的性价比高于重排。**

### 7.4 优化四：查询改写

用户的问题往往不是好的检索 query。三个常用手法：

1. **口语转书面**：让便宜模型先把"这玩意儿咋用啊"改写成"使用方法、操作步骤"再检索。
2. **多查询扩展**：一个问题改写成 3 个不同角度，各检索一遍，结果合并去重。召回率提升明显。
3. **HyDE**：让模型先凭空写一段"假如我知道答案，答案长这样"的假答案，用假答案去检索。因为**假答案的措辞更接近文档的措辞**，比原问题命中率高。

```python
REWRITE_PROMPT = """把用户的口语化问题改写成 3 个适合检索的短查询，每行一个，不要编号，不要解释。
问题：{q}"""
```

**注意**：查询改写会多一次模型调用，增加延迟和成本。**只有在"经常检索不到"时才用。**

### 7.5 优化五：小块检索、大块喂给模型

这是很实用的一招，解决矛盾：**检索要小块（精准），生成要大块（上下文完整）**。

做法：给每个小块记录它属于哪个父块（比如把父块 id 写进元数据）。检索命中小块后，**不把小块塞给模型，而是把它的父块塞进去**。

在你的 `splitter.py` 基础上扩展一句就行：先按 1200 字切父块，再在父块内部按 300 字切子块，子块元数据里带上 `parent_id`。检索用子块向量，喂模型用 `parent_id` 对应的父块文本。

---

## 第 8 章 里程碑 6：工程化与评估

### 8.1 做个小界面（可选）

最省力的选择是 Streamlit，一个文件搞定：

```powershell
uv pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
# app.py
import streamlit as st
from ask import ask  # 把上面 ask.py 的打印逻辑改成 return 更合适

st.title("我的知识库问答")
q = st.text_input("问点什么：")
if q:
    with st.spinner("检索并生成中..."):
        st.markdown(ask(q))
```

```powershell
streamlit run app.py
```

### 8.2 建立评估集（**最重要的一步，别跳过**）

没有评估集，你所有的"优化"都是玄学：改了 chunk_size 感觉变好了？可能只是这次问题恰好对上了。

**做法**：从你的资料里挑 **20~30 个问题**，人工标注"正确答案应该出自哪个文件（哪一页）"，存成 `eval_set.jsonl`：

```jsonl
{"question": "XXX 的保修期是多久？", "expect_source": "手册.pdf", "expect_page": 12}
{"question": "项目验收流程有几步？", "expect_source": "流程.md", "expect_page": 0}
```

然后写个脚本算两个指标：

- **Hit@k（命中率）**：期望来源有没有出现在检索结果的前 k 条里。**这是你最该盯的指标**——它直接反映检索质量。
- **MRR（平均倒数排名）**：命中那条排在第几位，越靠前分越高（第 1 名得 1 分，第 2 名得 0.5 分……）。

```python
# eval.py（骨架）
import json
from src.config import INDEX_PATH, CHUNK_META_PATH
from src.embedder import Embedder
from src.vector_store import FaissStore

def hit_at_k(hits, expect_source, expect_page=None):
    for score, meta in hits:
        if meta["source"] == expect_source and (not expect_page or meta["page"] == expect_page):
            return True
    return False

embedder = Embedder()
store = FaissStore(INDEX_PATH, CHUNK_META_PATH).load()

cases = [json.loads(l) for l in open("eval_set.jsonl", encoding="utf-8") if l.strip()]
hit = 0
for c in cases:
    hits = store.search(embedder.encode_query(c["question"]), k=5)
    ok = hit_at_k(hits, c["expect_source"], c.get("expect_page"))
    hit += ok
    print(("OK  " if ok else "MISS"), c["question"])

print(f"\nHit@5 = {hit / len(cases):.1%}")
```

**有了这个数字，你才敢说"我换了个 embedding 模型，效果从 65% 提到 82%"。** 这才是工程，其余都是感觉。

### 8.3 什么时候该上框架

当你遇到下面任一情况，再考虑 LlamaIndex / LangChain：
- 需要接入十几种数据源（网页、Notion、数据库）
- 需要复杂的 agent 编排（检索 → 判断 → 再检索 → 调用工具）
- 团队协作，需要统一的抽象和文档

**不是你一开始不会用，是你现在还不需要。** 而且有了第 8.2 节的评估集，你换成框架之后还能客观对比有没有变好——这才是升级的正确姿势。

### 8.4 想更进一步的典型方向

| 方向 | 具体做法 |
| --- | --- |
| 结构化知识 | 表格类内容单独走 SQL 查询，不要塞进向量库 |
| 多模态 | 图表/截图用多模态模型生成图片描述，把描述入向量库 |
| 增量索引 | 换成支持删除的库（Chroma / Qdrant），或用 `IDMap` 包装 FAISS |
| 权限控制 | 元数据加 `dept` / `level` 字段，检索时按用户身份过滤 |
| 效果监控 | 记录每次问答的检索结果与用户反馈，坏案例回流到评估集 |

---

## 第 9 章 本机高频坑清单

这些是在你这台机器上真会碰到的，按遇到概率排序。

---

### 坑 0 🔴 新增：torch 被装成「缝合怪」（本机遇到概率最高，而且已经踩过一次）

**症状**：`import torch` 直接崩，但**装包过程看起来一切正常**。

```text
ImportError: cannot import name 'thread_safe_generator' from 'torch.random'
```

**成因**：**同一个包，用两个不同的索引源分两次装。**
典型触发就是本文档 §2.3 原文的错误写法 —— 先用 PyTorch 官方源装了 CPU 版 torch，
再用清华源装 `requirements.txt`，导致 torch 被**重新解析成另一个版本**覆盖上去，
覆盖不彻底，新旧文件混在同一个 `torch/` 目录里。

**为什么难发现**：`uv pip install` 会打印 `Checked 10 packages in 449ms`，
`pip list` 也显示包都在 —— **但 `import` 就是起不来**。
它们只检查"依赖是否满足"，不检查"文件是否自洽"。

**三步查出缝合怪**：

| 检查 | 缝合怪特征 |
| --- | --- |
| `torch/version.py` 里的 `__version__` | 与元数据目录名**对不上**（例如文件写 `2.9.1+cpu`，目录却叫 `torch-2.14.0.dist-info`） |
| `torch/` 里各文件的**修改时间** | 明显分成**两批不同时间**（来自两个不同的 wheel） |
| 有没有 `functorch/`、`torchgen/` 目录 | 新版 torch 已删除这两个目录，**还在就说明混了旧文件** |

**解法**：把 `torch/`、`functorch/`、`torchgen/`、`torch-*.dist-info` 全部清掉，
然后用 §2.3 那条**唯一的单条命令**重装。

**预防（最重要）**：**永远只用一条 `uv pip install` 装完所有依赖，并用 `-c constraints.txt` 锁版本。**

---

### 坑 0.1 🔴 新增：torch 下载慢到怀疑人生

`torch 2.6.0+cpu` 的 wheel 约 **197MB**，且**只存在于 `download.pytorch.org`（境外）**。

| 场景 | 实测速度 | 下完 197MB 需要 |
| --- | --- | --- |
| 国内直连 | 0.1~0.5 MB/s | **约 70 分钟**（整条命令实测跑了 72 分钟） |
| 开代理（TUN 模式，全局生效） | 5.9~18.7 MB/s | **两三分钟** |

三个必须知道的点：
1. **必须后台跑** —— 前台跑会被系统掐断，表现是"命令没输出就结束了"，让你误判成失败；
2. **后台跑时看不到进度条** —— uv 的输出被重定向后就不画进度条了，**别以为它卡死了**；想看活性可以用进程 I/O 计数采样；
3. 清华源上的 torch 是 **CUDA 版（2GB+）**，更慢 —— 所以 **CPU 版躲不掉这个境外源**。

---

### 坑 1：`load_dotenv()` 不覆盖已存在的环境变量

`python-dotenv` **默认不覆盖**系统里已有的同名变量。你的 `HF_HOME` 是系统级设好的，所以**在 `.env` 里写 `HF_HOME=...` 完全无效**，还会误导你以为改了。

- 本项目因为直接传本地模型路径，绕过了这个问题，不需要动 `HF_HOME`。
- 真要临时改，就在 PowerShell 里 `$env:HF_HOME = "..."`，只对当前窗口生效。

### 坑 2：模型下载"成功"但文件 0 字节

国内直连 HuggingFace 常见症状：进度条走完了，文件是空的，然后报错。

- **首选方案：用本地目录**（本指南就是这么做的）。
- 次选：`$env:HF_ENDPOINT = "https://hf-mirror.com"`，然后重跑。
- 再不行：在浏览器/下载器里手动下 `pytorch_model.bin`（或 `model.safetensors`）、`config.json`、`vocab.txt`、`tokenizer.json` 到同一个文件夹，用本地路径加载。

### 坑 3：FAISS 装不上 / import 报错

主要是 numpy 2.x 与老版本 faiss-cpu 的 ABI 冲突。

```powershell
uv pip install -U faiss-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

还不行就退一档：`uv pip install "numpy<2"`。

### 坑 4：PyMuPDF 抽不出文字（扫描版 PDF）

`page.get_text()` 返回空白，说明这 PDF 是**图片扫描件**，没有文字层。

- 判断：抽出来的文本长度远小于页数 × 100。
- 解法：走 OCR。用 RapidOCR（轻量、中文好）或 PaddleOCR 把每页图片识别成文字，再进流水线。
- **不要**对着空文本反复调 chunk_size，那不是切分问题。

### 坑 5：中文编码

- 所有文件读写都显式写 `encoding="utf-8"`。
- `json.dumps` 要加 `ensure_ascii=False`，否则中文变成 `\u4e2d\u6587`，文件难读（功能不受影响）。
- Windows 控制台如果打印中文乱码，在 PowerShell 里执行 `chcp 65001`。

### 坑 6：向量 sqlite/faiss 文件被占用

`faiss.write_index` 时如果索引文件正被另一个进程读，会报权限错误。跑 `build_index.py` 前先确认没有别的脚本在跑。

### 坑 7：切分参数拍脑袋

`CHUNK_SIZE=500` 是起点不是真理。**一定要用第 8.2 节的评估集对比 300 / 500 / 800 三档的 Hit@5**，用数据选参数。

### 坑 8：问答答不出来就先怪模型

90% 的情况是**检索没捞到正确资料**。排查顺序永远是：

1. 先看检索结果（`ask.py` 已经把 Top-K 打印出来了）——正确资料的来源在不在里面？
2. 不在 → 检索问题（换 embedding / 加混合检索 / 调 chunk / 加查询改写）。
3. 在，但答案还是错 → prompt 问题（加强约束 / 降 temperature / 减少 K）。

**这个顺序能帮你省掉大量在模型上瞎折腾的时间。**

---

## 第 10 章 学习路线

### 建议的推进节奏

```text
第 3 章  模最小 RAG        -> 理解"检索就是向量点积"
第 4-5 章 解析/切分/建库    -> 理解"数据质量决定上限"
第 6 章  接大模型          -> 拿到第一个能用的成品
第 8.2 章 建评估集          -> 从"感觉好用"变成"知道多好用"
第 7 章  检索优化          -> 用数据驱动着调，而不是拍脑袋
第 8 章  工程化            -> 界面、增量、监控
```

**不要跳第 8.2 章。** 很多人做完第 6 章就急着上 LangChain、上 Milvus、上各种花哨技术，结果做出来的东西到底好不好，自己也不知道。评估集是你从"玩具"跨到"工程"的唯一分水岭。

### 记住一句话

> **RAG 的效果，20% 取决于大模型，80% 取决于你的文档解析、切分和检索策略。**

模型换来换去提升有限，把切分和检索打磨好，效果能翻倍。

---

## 附：文件清单速查

| 文件 | 作用 | 属于哪条链路 |
| --- | --- | --- |
| `constraints.txt` | ⚠️ **锁定 torch 版本，必须保留**；装依赖时要 `-c` 带上（见 §2.3） | - |
| `check_env.py` | 环境自检 | - |
| `mini_rag.py` | 最小演示（可删） | - |
| `try_split.py` | 验证解析切分（可删） | 离线 |
| `src/config.py` | 全部配置 | 两条都用 |
| `src/loader.py` | 文档解析 | 离线 |
| `src/splitter.py` | 文本切分 | 离线 |
| `src/embedder.py` | 文本转向量 | 两条都用 |
| `src/vector_store.py` | 向量存取 | 两条都用 |
| `src/retriever.py` | 混合检索（第 7 章） | 在线 |
| `src/llm.py` | 调大模型 | 在线 |
| `build_index.py` | 建索引入口 | 离线 |
| `ask.py` | 问答入口 | 在线 |
| `eval.py` | 评估检索质量 | 离线评估 |
