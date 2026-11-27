"""持股的「類型」（核心／交易）與「出場規則」：隔日交易訊號與盤中監控共用。

出場規則（出場線 = 均線 ×（1 − 緩衝））：
  13MA 加 1% 緩衝         收盤跌破 13MA × 0.99 → 全部賣出
  60MA 加 2% 緩衝         收盤跌破 60MA × 0.98 → 全部賣出
  120MA 加 2% 緩衝        收盤跌破 120MA × 0.98 → 全部賣出
  60MA 減半、120MA 出清   跌破 60MA × 0.98 → 賣一半；跌破 120MA × 0.98 → 全部賣出
類型：
  核心  預設 60MA 加 2% 緩衝；不套用「成本 − 2×ATR14」硬停損與 2R 停利（長抱不被短線波動洗出場），
        跌破 13MA 只提醒。
  交易  預設 13MA 加 1% 緩衝；保留硬停損與 2R 先賣一半。
自動建議（2009 年以來的含息日線，四項都通過才建議「核心」）：
  年化波動 ≤ 30%、收在年線（250MA）上方的時間 ≥ 70%、最長套牢（前高到再創新高）≤ 3.5 年、
  單年跌超過 30% 的次數 ≤ 1。歷史不足 5 年一律建議「交易」。
依據是 2009～2026 年的回測（9 檔持股：13MA 跌破就賣在每一檔都明顯輸給長抱），不保證未來，不構成投資建議。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

CORE, TRADE = "核心", "交易"
TYPES = [CORE, TRADE]
RULES = {
  "13MA 加 1% 緩衝": {"ma": 13, "buf": 0.01},
  "60MA 加 2% 緩衝": {"ma": 60, "buf": 0.02},
  "120MA 加 2% 緩衝": {"ma": 120, "buf": 0.02},
  "60MA 減半、120MA 出清": {"ma": 60, "buf": 0.02, "ma2": 120, "buf2": 0.02},
}
DEFAULT_RULE = {CORE: "60MA 加 2% 緩衝", TRADE: "13MA 加 1% 緩衝"}
MAX_MA = 120

TH_VOL, TH_ABOVE, TH_UNDER, TH_BADYEARS = 30.0, 70.0, 3.5, 1


# ---------- 股性分類 ----------
def _long_history(code: str) -> pd.Series:
  """2009 年以來的含息還原日線收盤；Yahoo 偶爾沒還原分割（單日跌超過 60%），自行還原"""
  for suf in (".TW", ".TWO"):
    try:
      j = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suf}",
                       params={"period1": 1230768000, "period2": int(time.time()), "interval": "1d",
                               "events": "div,split"},
                       headers={"User-Agent": "Mozilla/5.0"}, timeout=20).json()["chart"]["result"]
    except Exception:  # noqa: BLE001
      continue
    if not j:
      continue
    r = j[0]
    adj = (r.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not adj:
      continue
    s = pd.Series(adj, index=pd.to_datetime(r["timestamp"], unit="s").normalize(), dtype="float64").dropna()
    s = s[~s.index.duplicated()]
    for d, v in s.pct_change()[lambda x: x < -0.6].items():
      s[s.index < d] = s[s.index < d] / round(1 / (1 + v))
    if len(s) > 250:
      return s
  return pd.Series(dtype="float64")


@st.cache_data(ttl=86400, show_spinner="計算持股股性（2009 年以來日線）…")
def profile(code: str) -> dict:
  """四項股性指標與建議類型"""
  c = _long_history(code)
  if len(c) < 250 * 5:
    return {"代號": code, "建議類型": TRADE, "依據": "歷史不足 5 年（或抓不到資料），建議當交易持股"}
  yrs = (c.index[-1] - c.index[0]).days / 365.25
  dd = c / c.cummax() - 1
  under = (dd < 0).astype(int)
  longest = under.groupby((under.diff() != 0).cumsum()).sum().max() / 250
  yearly = c.resample("YE").last().pct_change().dropna()
  vol = float(c.pct_change().std() * np.sqrt(250) * 100)
  above = float((c > c.rolling(250).mean()).iloc[250:].mean() * 100)
  bad = int((yearly < -0.3).sum())
  checks = {"年化波動": vol <= TH_VOL, "年線上方": above >= TH_ABOVE,
            "最長套牢": longest <= TH_UNDER, "大跌年數": bad <= TH_BADYEARS}
  fails = [k for k, ok in checks.items() if not ok]
  return {"代號": code, "建議類型": CORE if not fails else TRADE,
          "年化報酬%": round(((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
          "年化波動%": round(vol, 0), "年線上方%": round(above, 0), "最長套牢(年)": round(float(longest), 1),
          "單年跌逾30%": bad, "最大跌幅%": round(float(dd.min()) * 100, 0),
          "依據": "四項都通過" if not fails else "未通過：" + "、".join(fails)}


def effective(hold: pd.DataFrame, suggest: bool = True) -> pd.DataFrame:
  """補上生效的類型與出場規則：類型空白 → 用股性建議；出場規則空白 → 用類型的預設"""
  h = hold.copy()
  for col in ("類型", "出場規則"):
    if col not in h:
      h[col] = ""
  h["類型"] = h["類型"].fillna("").astype(str)
  h["出場規則"] = h["出場規則"].fillna("").astype(str)
  src = []
  for i, r in h.iterrows():
    s = "自訂"
    if r["類型"] not in TYPES:
      h.at[i, "類型"] = profile(r["代號"])["建議類型"] if suggest else TRADE
      s = "自動建議"
    if r["出場規則"] not in RULES:
      h.at[i, "出場規則"] = DEFAULT_RULE[h.at[i, "類型"]]
    src.append(s)
  h["設定來源"] = src
  return h


# ---------- 規則判斷 ----------
def exit_lines(ma: dict, rule: str) -> dict:
  """ma：{13: 值, 60: 值, 120: 值}。回傳出場線（缺均線時為 None）"""
  r = RULES[rule]
  out = {"line": ma.get(r["ma"]) * (1 - r["buf"]) if ma.get(r["ma"]) else None, "ma": r["ma"]}
  if "ma2" in r:
    out["line2"] = ma.get(r["ma2"]) * (1 - r["buf2"]) if ma.get(r["ma2"]) else None
    out["ma2"] = r["ma2"]
  return out


def judge(price: float, ma: dict, rule: str) -> tuple[str, str, float | None]:
  """回傳 (動作, 說明, 出場線)。動作：sell 全部賣出／half 賣一半／hold 續抱／na 均線資料不足"""
  ln = exit_lines(ma, rule)
  if ln["line"] is None:
    return "na", f"資料不足，算不出 {ln['ma']}MA", None
  if ln.get("line2") is not None and price < ln["line2"]:
    return "sell", f"跌破 {ln['ma2']}MA 出場線 {ln['line2']:,.2f}（{rule}）", ln["line2"]
  if price < ln["line"]:
    if "line2" in ln:
      return "half", f"跌破 {ln['ma']}MA 出場線 {ln['line']:,.2f}，先賣一半（已減半者續抱剩餘；{rule}）", ln["line"]
    return "sell", f"跌破 {ln['ma']}MA 出場線 {ln['line']:,.2f}（{rule}）", ln["line"]
  return "hold", f"守在 {ln['ma']}MA 出場線 {ln['line']:,.2f} 之上（距 {(price / ln['line'] - 1) * 100:+.2f}%；{rule}）", ln["line"]
