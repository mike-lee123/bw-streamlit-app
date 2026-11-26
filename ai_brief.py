"""第 26 艙「七、AI 研判」：把規則報告交給 Claude 寫成研判段落。

原則：燈號、名單、部位、停損全部由規則決定，AI 只負責解說，不能改動。
防呆：產生後檢查 AI 文中出現的數字是否都能在規則報告裡找到，對不上的列出警示。

金鑰：Streamlit secrets 的 ANTHROPIC_API_KEY（本機 .streamlit/secrets.toml，雲端在 App settings → Secrets），
或環境變數 ANTHROPIC_API_KEY。模型可用 secrets 的 ANTHROPIC_MODEL 覆寫，預設 claude-opus-5-5。
"""

from __future__ import annotations

import logging
import os
import re

import anthropic
import streamlit as st

_log = logging.getLogger(__name__)
DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """你是台股盤後的技術分析研究員。使用者會給你一份由固定規則產生的「每日綜合戰情總結報告」（Markdown），
請根據它寫一段研判，給個人投資人在收盤後閱讀。

規則：
- 報告裡的燈號、選股名單、部位上限、停損價是規則算出來的定案，你只能解說，不能更改、推翻或另外給買賣建議。
- 只能引用報告裡出現的數字；不要自行加入報告沒有的價格、新聞、基本面或總經資料。需要的數字報告裡沒有時，就說「報告未提供」。
- 台股慣例紅漲綠跌；USD/TWD 上漲代表台幣貶值；未平倉 PCR 高代表下方支撐強。
- 用台灣慣用的繁體中文與術語。

輸出格式（不要用 # 標題，小標用【】，可用 **粗體** 與條列）：
【綜合研判】2～3 句：這些訊號合起來代表什麼，市場處在什麼狀態。
【訊號矛盾】哪些訊號和多數方向相反、代表什麼；沒有就寫「無明顯矛盾」。
【情境推演】2～3 個情境（例如止跌、續跌、反彈），每個情境寫出要看哪個數字、到哪個價位才算確認（引用報告的「明日觀察」；下跌情境請引用「下檔支撐」表，寫出跌破哪個支撐後下一個看哪裡）。
【操作紀律】根據報告的風控與「五、隔日交易訊號」段落，逐一說明隔日要執行的買進／賣出／續抱訊號與理由，提醒部位與停損的執行重點（不得改動數字，也不得新增報告沒有的買賣標的）。
全文 400～700 字。最後一行固定寫：「以上由 AI 根據規則報告撰寫，僅供參考，不構成投資建議。」"""


def key_status() -> tuple[str | None, str]:
  """回傳 (金鑰, 找不到時的原因)。
  原因只給一般性的提示：畫面可能被他人看到，所以不列出其他 secrets 的名稱、也不顯示解析錯誤原文
  （TOML 錯誤訊息可能引用 secrets 檔的內容）。詳細錯誤只寫進伺服器 log。"""
  try:
    key = st.secrets.get("ANTHROPIC_API_KEY")
  except Exception as e:  # noqa: BLE001
    if type(e).__name__ == "StreamlitSecretNotFoundError":
      key, why = None, "找不到任何 secrets（本機沒有 .streamlit/secrets.toml，或雲端 Secrets 是空的）"
    else:
      _log.warning("secrets 解析失敗：%s", type(e).__name__)
      env = os.environ.get("ANTHROPIC_API_KEY")
      return (env, "") if env else (None, "Secrets 內容格式錯誤，無法解析（請檢查是否用半形雙引號、開頭沒有 #）")
  else:
    if key:
      key = str(key).strip()
      if not key.startswith("sk-ant-"):
        return None, "ANTHROPIC_API_KEY 有值但不是 sk-ant- 開頭，請確認貼上的是 Anthropic API 金鑰"
      return key, ""
    why = "Secrets 裡沒有 ANTHROPIC_API_KEY 這一行（若放在 [區段] 底下，請移到所有區段的上方）"
  env = os.environ.get("ANTHROPIC_API_KEY")
  return (env, "") if env else (None, why)


def api_key() -> str | None:
  return key_status()[0]


def model_name() -> str:
  try:
    return st.secrets.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
  except Exception:  # noqa: BLE001
    return DEFAULT_MODEL


@st.cache_data(ttl=86400, show_spinner=False)
def generate(report_md: str, model: str, _key: str) -> dict:
  """同一份報告當天只呼叫一次（快取 24 小時）；失敗會丟例外，例外不會被快取。"""
  client = anthropic.Anthropic(api_key=_key, timeout=180.0)
  resp = client.beta.messages.create(
    model=model,
    max_tokens=16000,
    system=SYSTEM_PROMPT,
    output_config={"effort": "medium"},
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",            # 安全分類器拒答時，伺服器自動改用備援模型重跑
    messages=[{"role": "user", "content": f"以下是今天的規則報告：\n\n{report_md}"}],
  )
  if resp.stop_reason == "refusal":
    raise RuntimeError("Claude 拒絕產生這份內容（refusal）。")
  text = "".join(b.text for b in resp.content if b.type == "text").strip()
  if not text:
    raise RuntimeError(f"Claude 沒有回傳文字（stop_reason={resp.stop_reason}）。")
  return {"text": text, "model": resp.model, "stop_reason": resp.stop_reason,
          "in_tok": resp.usage.input_tokens, "out_tok": resp.usage.output_tokens}


# ---------- 數字比對 ----------
_NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
  out = set()
  for m in _NUM.findall(text):
    s = m.replace(",", "").replace("−", "-").lstrip("+-")
    if s:
      out.add(s)
  return out


def _variants(n: str) -> set[str]:
  """報告數字的常見寫法：原樣、四捨五入到整數／1 位／2 位小數"""
  out = {n}
  try:
    f = float(n)
  except ValueError:
    return out
  for d in (0, 1, 2):
    out.add(f"{f:.{d}f}")
  return out


def unmatched_numbers(ai_text: str, report_md: str) -> list[str]:
  """AI 文中出現、但規則報告裡找不到的數字（忽略 ≤ 10 的小整數，例如「3 個情境」「5 日均」）"""
  known = set().union(*(_variants(n) for n in _numbers(report_md)))
  bad = []
  for n in sorted(_numbers(ai_text)):
    if n in known:
      continue
    if n.isdigit() and int(n) <= 10:
      continue
    bad.append(n)
  return bad


def error_text(e: Exception) -> str:
  if isinstance(e, anthropic.AuthenticationError):
    return "API 金鑰無效，請檢查 ANTHROPIC_API_KEY。"
  if isinstance(e, anthropic.PermissionDeniedError):
    return "這把金鑰沒有使用此模型的權限。"
  if isinstance(e, anthropic.NotFoundError):
    return f"找不到模型（{model_name()}），請檢查 ANTHROPIC_MODEL。"
  if isinstance(e, anthropic.RateLimitError):
    return "請求太頻繁或額度不足（429），請稍後再試。"
  if isinstance(e, anthropic.BadRequestError):
    return f"請求格式錯誤：{e.message}"
  if isinstance(e, anthropic.APIStatusError):
    return f"Anthropic 伺服器錯誤（{e.status_code}），請稍後再試。"
  if isinstance(e, anthropic.APIConnectionError):
    return "連不上 Anthropic API，請檢查網路。"
  if isinstance(e, RuntimeError):          # 本模組自己丟出的（拒答、空回應），訊息內容可控
    return str(e)
  _log.warning("AI 研判失敗：%r", e)        # 其他未預期錯誤只寫 log，畫面不顯示原文
  return f"未預期的錯誤（{type(e).__name__}），詳情請看伺服器 log。"
