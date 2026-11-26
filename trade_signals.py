"""第 26 艙「五、隔日交易訊號」與「📒 訊號戰績」。

全部是固定規則的機械輸出，不是投資建議：
  買進：入選名單 → 隔日開盤 ≤ 今收 +2% 才買（不追高）、掛單參考今收或拉回 13MA、停損 = 收盤 − 2×ATR14、
        停利 = 2R 先賣一半（R = 收盤 − 停損），其餘用 13MA 移動停利。
  賣出：逐檔檢查持股
        🔴 賣出     收盤跌破 13MA，或跌破停損（成本 − 2×ATR14）
        🟡 賣一半   獲利 ≥ 2R（R = 2×ATR14）→ 賣一半，其餘以 13MA 移動停利
        🟠 減碼     燈號的總部位上限被超過 → 等比例減到上限
        🟢 續抱     其他情況；13MA 是隔日的警戒價
  戰績：每次產生報告就把訊號記下來（同一天同一檔同一種訊號只記一次），之後回頭算 1／5／20 日後的報酬。

個人資料（持股、戰績紀錄）只存在本機 data/ 資料夾（已列入 .gitignore），不會進 GitHub。
Streamlit Cloud 重開機會清空本機檔案，雲端請用「下載／上傳 CSV」保存。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"
HOLDINGS_FILE = DATA_DIR / "holdings.csv"
LOG_FILE = DATA_DIR / "signal_log.csv"

OPEN_CHASE_LIMIT = 0.02     # 隔日開盤高於今收 2% 以上就不追
ATR_MULT = 2.0              # R = 2 × ATR14
TAKE_PROFIT_R = 2.0         # 獲利 2R 先賣一半
HOLD_COLS = ["代號", "成本", "股數"]
LOG_COLS = ["日期", "代號", "名稱", "訊號", "參考價", "停損", "燈號", "理由"]
HORIZONS = (1, 5, 20)


# ---------- 持股 ----------
def _clean_holdings(df: pd.DataFrame) -> pd.DataFrame:
  df = df.reindex(columns=HOLD_COLS).copy()
  df["代號"] = df["代號"].astype(str).str.strip().str.upper().str.replace(r"\.TWO?$", "", regex=True)
  df = df[df["代號"].str.fullmatch(r"[0-9A-Z]{4,6}")]
  df["成本"] = pd.to_numeric(df["成本"], errors="coerce")
  df["股數"] = pd.to_numeric(df["股數"], errors="coerce")
  return df.drop_duplicates("代號").reset_index(drop=True)


def load_holdings() -> pd.DataFrame:
  if HOLDINGS_FILE.exists():
    try:
      return _clean_holdings(pd.read_csv(HOLDINGS_FILE, dtype={"代號": str}))
    except Exception:  # noqa: BLE001
      pass
  return pd.DataFrame(columns=HOLD_COLS)


def save_holdings(df: pd.DataFrame) -> bool:
  try:
    DATA_DIR.mkdir(exist_ok=True)
    _clean_holdings(df).to_csv(HOLDINGS_FILE, index=False, encoding="utf-8-sig")
    return True
  except OSError:
    return False


# ---------- 買進訊號 ----------
def buy_signals(plan: pd.DataFrame, held: set[str]) -> pd.DataFrame:
  if plan is None or plan.empty:
    return pd.DataFrame()
  p = plan.copy()
  r = p["收盤"] - p["停損"]
  out = pd.DataFrame({
    "訊號": np.where(p["代號"].isin(held), "🟢 加碼", "🟢 買進"),
    "代號": p["代號"], "名稱": p["名稱"], "日期": p["日期"], "今收": p["收盤"],
    "開盤上限（不追）": (p["收盤"] * (1 + OPEN_CHASE_LIMIT)).round(2),
    "拉回掛單（13MA）": p["13MA"].round(2),
    "停損": p["停損"], "停利一半（2R）": (p["收盤"] + TAKE_PROFIT_R * r).round(2),
    "張數": p["張數"], "共振分數": p["共振分數"],
  })
  return out.reset_index(drop=True)


# ---------- 賣出訊號 ----------
def sell_signals(holdings: pd.DataFrame, rows: pd.DataFrame, regime_cap_pct: float, capital: float) -> pd.DataFrame:
  """holdings：代號／成本／股數；rows：同一批代號的掃描結果（收盤、13MA、ATR14、共振分數…）"""
  if holdings.empty:
    return pd.DataFrame()
  info = rows.set_index("代號") if not rows.empty else pd.DataFrame()
  out = []
  for _, h in holdings.iterrows():
    code = h["代號"]
    if code not in info.index:
      out.append({"訊號": "⚪ 無資料", "代號": code, "名稱": "", "今收": None, "13MA": None, "停損": None,
                  "損益%": None, "建議賣出股數": None, "理由": "抓不到行情或資料不足 80 天"})
      continue
    r = info.loc[code]
    close, ma13, atr = float(r["收盤"]), float(r["13MA"]), float(r["ATR14"])
    cost, shares = h["成本"], h["股數"]
    R = ATR_MULT * atr
    stop = cost - R if pd.notna(cost) else None
    pnl = (close / cost - 1) * 100 if pd.notna(cost) and cost > 0 else None
    if close < ma13:
      sig, why, qty = "🔴 賣出", f"收盤 {close:,.2f} 跌破 13MA {ma13:,.2f}（13MA 熔斷防線）", shares
    elif stop is not None and close < stop:
      sig, why, qty = "🔴 賣出", f"收盤 {close:,.2f} 跌破停損 {stop:,.2f}（成本 − 2×ATR14）", shares
    elif pd.notna(cost) and close >= cost + TAKE_PROFIT_R * R:
      sig, qty = "🟡 賣一半", (int(shares // 2) if pd.notna(shares) else None)
      why = f"獲利 {pnl:+.1f}% 已達 2R（{cost + TAKE_PROFIT_R * R:,.2f}），賣一半、其餘以 13MA {ma13:,.2f} 移動停利"
    else:
      sig, qty = "🟢 續抱", 0
      why = f"守在 13MA {ma13:,.2f} 之上（距 {(close / ma13 - 1) * 100:+.2f}%）；收盤跌破即賣出"
    out.append({"訊號": sig, "代號": code, "名稱": r["名稱"], "日期": r["日期"], "今收": round(close, 2), "13MA": round(ma13, 2),
                "停損": round(stop, 2) if stop is not None else None,
                "損益%": round(pnl, 2) if pnl is not None else None,
                "建議賣出股數": qty, "理由": why, "_市值": close * shares if pd.notna(shares) else 0.0})
  df = pd.DataFrame(out)
  if "_市值" not in df:
    df["_市值"] = 0.0
  df["_市值"] = df["_市值"].fillna(0.0)

  # 🟠 總部位上限：賣出／賣一半之後剩下的市值若仍超過上限，對「續抱」的持股等比例減碼
  budget = capital * regime_cap_pct / 100
  keep = df["訊號"] == "🟢 續抱"
  remain = df.loc[keep, "_市值"].sum() + (df.loc[df["訊號"] == "🟡 賣一半", "_市值"] / 2).sum()
  if remain > budget and df.loc[keep, "_市值"].sum() > 0:
    cut = min(1.0, (remain - budget) / df.loc[keep, "_市值"].sum())
    for i in df.index[keep]:
      sh = holdings.set_index("代號").loc[df.at[i, "代號"], "股數"]
      if pd.notna(sh) and sh > 0:
        df.at[i, "訊號"] = "🟠 減碼"
        df.at[i, "建議賣出股數"] = int(np.ceil(sh * cut))
        df.at[i, "理由"] += f"；總持股 {remain:,.0f} 元超過燈號上限 {regime_cap_pct:.0f}%（{budget:,.0f} 元），等比例減碼 {cut * 100:.0f}%"
  # 沒填股數時改用文字（全部／一半），有填則顯示整數股數
  qty = pd.to_numeric(df["建議賣出股數"], errors="coerce").round()
  df["建議賣出股數"] = [
    (f"{int(q):,} 股" if q > 0 else "—") if pd.notna(q)
    else {"🔴 賣出": "全部", "🟡 賣一半": "一半"}.get(sig, "—")
    for q, sig in zip(qty, df["訊號"])]
  return df.drop(columns="_市值")


# ---------- 戰績紀錄 ----------
def load_log() -> pd.DataFrame:
  if LOG_FILE.exists():
    try:
      return pd.read_csv(LOG_FILE, dtype={"代號": str})
    except Exception:  # noqa: BLE001
      pass
  return pd.DataFrame(columns=LOG_COLS)


def append_log(date: str, regime: str, buys: pd.DataFrame, sells: pd.DataFrame) -> int:
  """把今天的訊號寫進紀錄；回傳新增筆數（同日同檔同訊號不重複）。續抱與無資料不記。
  日期用該檔股票自己的最新資料日（大盤資料若落後一天，也不會把訊號記錯天）；沒有才用 date。"""
  new = []
  for _, b in (buys if buys is not None else pd.DataFrame()).iterrows():
    new.append({"日期": b.get("日期") or date, "代號": b["代號"], "名稱": b["名稱"], "訊號": "買進", "參考價": b["今收"],
                "停損": b["停損"], "燈號": regime, "理由": f"共振 {b['共振分數']} 分"})
  for _, s in (sells if sells is not None else pd.DataFrame()).iterrows():
    kind = {"🔴 賣出": "賣出", "🟡 賣一半": "賣一半", "🟠 減碼": "減碼"}.get(s["訊號"])
    if kind:
      new.append({"日期": s.get("日期") or date, "代號": s["代號"], "名稱": s["名稱"], "訊號": kind, "參考價": s["今收"],
                  "停損": s["停損"], "燈號": regime, "理由": s["理由"]})
  if not new:
    return 0
  log = load_log()
  merged = pd.concat([log, pd.DataFrame(new)], ignore_index=True) if not log.empty else pd.DataFrame(new)
  merged = merged.drop_duplicates(["日期", "代號", "訊號"], keep="first")
  added = len(merged) - len(log)
  if added:
    try:
      DATA_DIR.mkdir(exist_ok=True)
      merged.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    except OSError:
      return 0
  return added


def track_record(log: pd.DataFrame, fetch) -> tuple[pd.DataFrame, pd.DataFrame]:
  """每筆訊號在 1／5／20 個交易日後的報酬。
  買進：報酬 = 之後收盤 ÷ 參考價 − 1，並檢查 20 日內是否碰到停損。
  賣出類：「避開跌幅」= 參考價 ÷ 之後收盤 − 1（賣出後股價下跌為正＝賣對）。"""
  if log.empty:
    return pd.DataFrame(), pd.DataFrame()
  rows = []
  prices = {}
  for _, s in log.iterrows():
    code = str(s["代號"])
    if code not in prices:
      prices[code] = fetch(code, range_str="1y")
    px = prices[code]
    rec = {"日期": s["日期"], "代號": code, "名稱": s["名稱"], "訊號": s["訊號"], "參考價": s["參考價"]}
    if px is None or px.empty:
      rows.append(rec)
      continue
    after = px[px.index > pd.Timestamp(s["日期"])]
    buy = s["訊號"] == "買進"
    for n in HORIZONS:
      if len(after) >= n:
        p = float(after["Close"].iloc[n - 1])
        ret = (p / float(s["參考價"]) - 1) * 100
        rec[f"{n}日%"] = round(ret if buy else -ret, 2)
    if buy and pd.notna(s.get("停損")) and len(after):
      rec["20日內觸停損"] = bool((after["Low"].head(20) < float(s["停損"])).any())
    rows.append(rec)
  detail = pd.DataFrame(rows)
  summ = []
  for kind, g in detail.groupby("訊號"):
    row = {"訊號": kind, "筆數": len(g)}
    for n in HORIZONS:
      col = f"{n}日%"
      if col in g and g[col].notna().any():
        v = g[col].dropna()
        row[f"{n}日 已滿筆數"] = len(v)
        row[f"{n}日 勝率%"] = round((v > 0).mean() * 100, 1)
        row[f"{n}日 平均%"] = round(v.mean(), 2)
    summ.append(row)
  note = "買進看報酬；賣出／賣一半／減碼看「避開跌幅」（賣出後股價下跌為正，代表賣對）。"
  summary = pd.DataFrame(summ)
  summary.attrs["note"] = note
  return summary, detail
