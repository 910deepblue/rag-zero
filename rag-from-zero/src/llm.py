# -*- coding: utf-8 -*-
"""
llm.py —— 大模型调用封装（RAG 的最后一环）

职责：把【用户问题 + 检索到的原文段落】拼成 messages 发给大模型，拿回答案文本。
特点：走 OpenAI 兼容协议，所以既可以用官方 OpenAI，也可以换成阿里云百炼(DashScope)、
      智谱、DeepSeek 等任何提供 /compatible-mode 的服务 —— 只改 .env 里的 base_url 和模型名即可。
"""

import os                          # 读环境变量(密钥、模型名都从这里来)

from dotenv import load_dotenv     # 从 .env 文件把配置加载进 os.environ
from openai import OpenAI          # 用 OpenAI 官方 SDK 访问兼容接口


# 加载 .env。
# 注意两点(都很容易踩)：
#   1) 它默认【从当前工作目录往上找】.env，所以脚本必须在项目根目录跑，
#      否则读不到文件、api_key 会是 None（和 build_index.py 的目录要求是同一个道理）。
#   2) 它默认【不覆盖】已经存在的环境变量 —— 系统里若已有同名变量，.env 里写的会被忽略。
load_dotenv()

# 模块级创建客户端：整个进程只建一次，复用底层 HTTP 连接池。
# ⚠️ 注意这是"import 时就执行"的：如果 LLM_API_KEY 没配好(名字不匹配或值为空)，
#    那么在 import 这个模块的瞬间就会抛 OpenAIError（Missing credentials），
#    而不是等到真正调用 chat() 才报错 —— 排查时要认准这个时间点。
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),        # 密钥：绝不写死在代码里，只从环境变量取
    base_url=os.getenv("LLM_BASE_URL"),      # 兼容模式地址，例：https://api.deepseek.com
)

# 模型名的**默认值兜底**写法：环境变量里有就用它，没有就退回 "deepseek-v4-flash"。
# 好处是 .env 里漏配这一项时程序仍能跑，坏处是"配错了也不报错"，排查时先 print 一下确认。
MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")



def chat(messages: list[dict], temperature: float = 0.2) -> str:
    """发一轮对话，返回模型的回答文本

    messages 是标准 OpenAI 格式的列表，每项形如 {"role": "user", "content": "..."}；
    role 取值：system(设定人设/约束) / user(用户提问) / assistant(历史回答)。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,   # 问答要稳，温度调低
    )
    # OpenAI 兼容接口的统一返回路径：choices[0] 是最优候选，message.content 是文本内容。
    # （若模型只返回工具调用、或内容被安全策略拦截，content 可能是 None —— 见下面的提醒）
    return resp.choices[0].message.content
