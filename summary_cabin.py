"""第 26 艙「📋 每日綜合戰情總結艙」。

把各艙的結論用固定規則（不接 AI）合成一頁總結報告：
  ① 大盤燈號：8 票投票（加權均線排列、13MA 多空開關、加權 MACD 動能、股匯象限、櫃買（富櫃50 ETF）均線排列、費半趨勢、VIX、期交所選擇權未平倉 PCR）
  ② 選股名單：沿用「🌐 全策略總合指揮艙」的共振分數，再加 13MA 防線與「📡 雷達艙」的法人籌碼確認
  ③ 進場時機：日線跳空缺口、爆量
  ④ 風控：依燈號決定總部位上限、單筆風險與最低共振門檻；停損 = 收盤 − 2×ATR14
  ⑤ 明日觀察：每張空方（或多方）票要翻轉的條件
報告可下載 Markdown / HTML。app.py 內的策略函式以 deps 傳入（app.py 是腳本，不能被 import）。

PCR 來自期交所（taifex_pcr.py）、VIX 來自 Yahoo ^VIX，都是真實數據。全商品艙的夜盤、回測艙的績效是示範用的固定值，本艙不採用。
期現價差需要永豐連線，第一版不納入投票。
"""

from __future__ import annotations

import html
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import exit_rules as er
import trade_signals as ts
from taifex_pcr import PCR_BEAR, PCR_BULL, pcr_signal, taifex_pcr
from fx_cabin import DN, UP, _fx_layout, _fx_macd_panel, fx_add_indicators, fx_build_link, fx_fetch, fx_pct_change

# ---------- 規則參數 ----------
REGIMES = {
  # 燈號: (總部位上限%, 單筆風險%, 最低共振分數)
  "🟢 偏多": (80, 1.0, 3),
  "🟡 中性": (50, 0.75, 4),
  "🔴 偏空": (20, 0.5, 4),
}
REGIME_THRESHOLD = 3      # 合計 ≥ +3 偏多、≤ −3 偏空
CORR_STRONG = 0.3         # 股匯相關高於此值，匯率票才計分
VIX_CALM, VIX_FEAR = 18.0, 25.0
ATR_STOP_MULT = 2.0
DEFAULT_GROUP = "半導體與 AI 權值核心"
VOTE_ICON = {1: "🔴 +1", -1: "🟢 −1", 0: "⚪ 0"}   # 台股慣例：紅多綠空


# ---------- 指標小工具 ----------
def _trend(row) -> int:
  if row.Close > row.MA20 > row.MA60:
    return 1
  if row.Close < row.MA20 < row.MA60:
    return -1
  return 0


TREND_TXT = {1: "多頭排列", -1: "空頭排列", 0: "盤整"}


def _ma13_state(df: pd.DataFrame) -> tuple[int, float, float]:
  """13MA 艙的多空狀態：收盤 vs 13MA、KD 的 K 值 vs 50、MACD 柱 vs 0，三者同向才成立。回傳 (狀態, K, MACD 柱)"""
  c = df["Close"]
  ma13 = c.rolling(13).mean()
  low9, high9 = df["Low"].rolling(9).min(), df["High"].rolling(9).max()
  rsv = ((c - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
  k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
  last = df.index[-1]
  above, k_now, o = c[last] > ma13[last], float(k[last]), float(df["HIST"][last])
  if above and k_now > 50 and o > 0:
    return 1, k_now, o
  if (not above) and k_now < 50 and o < 0:
    return -1, k_now, o
  return 0, k_now, o


FIB_RATIOS = (0.382, 0.5, 0.618)


def _unfilled_gaps(df: pd.DataFrame, lookback: int = 120) -> tuple[tuple | None, tuple | None]:
  """最近一個未回補的向上缺口（現價下方＝支撐）與向下缺口（現價上方＝壓力）。回傳 (日期, 缺口上緣, 缺口下緣)"""
  d = df.tail(lookback + 1)
  hi, lo = d["High"].to_numpy(), d["Low"].to_numpy()
  up = dn = None
  for i in range(len(d) - 1, 0, -1):
    if up is None and lo[i] > hi[i - 1] and lo[i:].min() > hi[i - 1]:     # 向上缺口：之後低點沒跌回缺口下緣
      up = (d.index[i].date().isoformat(), float(lo[i]), float(hi[i - 1]))
    if dn is None and hi[i] < lo[i - 1] and hi[i:].max() < lo[i - 1]:     # 向下缺口：之後高點沒漲回缺口上緣
      dn = (d.index[i].date().isoformat(), float(lo[i - 1]), float(hi[i]))
    if up and dn:
      break
  return up, dn


def key_levels(df: pd.DataFrame) -> pd.DataFrame:
  """加權指數支撐／壓力表（全部由規則計算）：近期高低點、布林通道、MA20／60／120、費波那契回撤、未回補缺口"""
  c = float(df.Close.iloc[-1])
  rows = [
    ("20 日最低", df.Low.tail(20).min()), ("60 日最低", df.Low.tail(60).min()),
    ("20 日最高", df.High.tail(20).max()),
    ("布林下軌（MA20−2σ）", df.BB_dn.iloc[-1]), ("布林上軌（MA20+2σ）", df.BB_up.iloc[-1]),
    ("MA20", df.MA20.iloc[-1]), ("MA60", df.MA60.iloc[-1]),
    ("MA120", df.Close.rolling(120).mean().iloc[-1]),
  ]
  # 費波那契：60 日內最高點，對應它之前 250 日內的波段低點 → 這段漲幅的回撤位
  hi_idx = df.High.tail(60).idxmax()
  swing_hi = float(df.High[hi_idx])
  swing_lo = float(df.Low[:hi_idx].tail(250).min())
  for r in FIB_RATIOS:
    rows.append((f"費波那契 {r:.3f} 回撤（{swing_lo:,.0f}→{swing_hi:,.0f}）", swing_hi - (swing_hi - swing_lo) * r))
  up, dn = _unfilled_gaps(df)
  if up:
    rows.append((f"未回補向上缺口（{up[0]}，{up[2]:,.0f}～{up[1]:,.0f}）", up[2]))
  if dn:
    rows.append((f"未回補向下缺口（{dn[0]}，{dn[2]:,.0f}～{dn[1]:,.0f}）", dn[1]))
  out = pd.DataFrame(rows, columns=["類型", "價位"]).dropna()
  out["價位"] = out["價位"].astype(float).round(0)
  out = out.groupby("價位", as_index=False, sort=False)["類型"].agg("／".join)   # 同價位（例如 20 日與 60 日最低相同）合併標示
  out["位置"] = np.where(out["價位"] < c, "支撐", "壓力")
  out["距現價%"] = ((out["價位"] / c - 1) * 100).round(2)
  now = pd.DataFrame([{"類型": "▶ 現價", "價位": round(c, 0), "位置": "—", "距現價%": 0.0}])
  out = pd.concat([out, now]).sort_values("價位", ascending=False).reset_index(drop=True)
  out["價位"] = out["價位"].map("{:,.0f}".format)
  return out[["位置", "價位", "距現價%", "類型"]]


def _vote(name, source, score, reading, flip):
  return {"項目": name, "來源艙": source, "票": int(score), "讀數": reading, "翻轉條件": flip}


# ---------- ① 大盤投票 ----------
@st.cache_data(ttl=1800, show_spinner="計算大盤燈號…")
def market_votes() -> tuple[list[dict], dict]:
  votes, info = [], {}

  twii = fx_add_indicators(fx_fetch("^TWII", "1y"))
  last, prev = twii.iloc[-1], twii.iloc[-2]
  info["date"] = twii.index[-1].date().isoformat()
  info["twii"] = float(last.Close)
  info["twii_chg"] = float(last.Close / prev.Close - 1) * 100
  info["twii_rsi"] = float(last.RSI)
  info["levels"] = key_levels(twii)

  t = _trend(last)
  votes.append(_vote("加權指數均線排列", "💱 台幣匯率 / 📈 現貨", t,
                     f"{TREND_TXT[t]}（收 {last.Close:,.0f}、MA20 {last.MA20:,.0f}、MA60 {last.MA60:,.0f}）",
                     f"收盤站回 MA20（{last.MA20:,.0f}）" if t <= 0 else f"收盤跌破 MA20（{last.MA20:,.0f}）"))

  s13, k13, osc13 = _ma13_state(twii)
  ma13 = float(twii.Close.rolling(13).mean().iloc[-1])
  info["k"] = k13
  votes.append(_vote("13MA 多空開關", "📐 13MA的魔力", s13,
                     {1: "多頭成立", -1: "空頭成立", 0: "多空未定"}[s13]
                     + f"（13MA {ma13:,.0f}、K {k13:.1f}、MACD 柱 {osc13:+.1f}）",
                     f"收盤站上 13MA（{ma13:,.0f}）且 K>50、MACD 柱翻正" if s13 <= 0
                     else f"收盤跌破 13MA（{ma13:,.0f}）"))

  h_now, h_prev = float(last.HIST), float(prev.HIST)
  if h_now > 0 and h_now > h_prev:
    m, mt = 1, "多方動能增強"
  elif h_now < 0 and h_now < h_prev:
    m, mt = -1, "空方動能增強"
  else:
    m, mt = 0, "多方動能減弱" if h_now >= 0 else "空方動能減弱"
  votes.append(_vote("加權 MACD 動能", "🎯 MACD強化版", m, f"{mt}（柱 {h_now:+.1f}）",
                     "柱狀體由長轉短（空方動能減弱）" if m < 0 else "柱狀體連續縮短則轉中性"))

  link = fx_build_link("^TWII", "1y", 60)
  corr = float(link.CORR.iloc[-1])
  twd20 = -fx_pct_change(link.FX, 20)          # 正＝台幣升值
  stk20 = fx_pct_change(link.Close, 20)
  fx_now = float(link.FX.iloc[-1])
  fx_ma20 = float(link.FX.rolling(20).mean().iloc[-1])
  info.update(corr=corr, twd20=twd20, stk20=stk20, usdtwd=fx_now)
  if pd.isna(corr) or corr <= CORR_STRONG:
    f, ft = 0, f"股匯連動弱（相關 {corr:+.2f}），不計分"
  elif twd20 > 0 and stk20 > 0:
    f, ft = 1, "資金行情：台幣升＋股市漲"
  elif twd20 < 0 and stk20 < 0:
    f, ft = -1, "股匯雙殺：台幣貶＋股市跌（外資撤出跡象）"
  else:
    f, ft = 0, "股匯分歧，方向不明"
  votes.append(_vote("股匯象限", "💱 台幣匯率×台股", f,
                     f"{ft}；20 日台幣 {twd20:+.2f}%、加權 {stk20:+.2f}%、相關 {corr:+.2f}",
                     f"USD/TWD 跌回 MA20（{fx_ma20:.3f}）以下，台幣止貶" if f < 0
                     else f"USD/TWD 站上 MA20（{fx_ma20:.3f}）則轉弱" if f > 0
                     else "相關 >0.3 且股匯同向才計分"))

  # Yahoo 已無櫃買指數（^TWOII 404），改用追蹤櫃買市場的 元大富櫃50（006201）代表中小型股
  for sym, name, src in (("006201.TWO", "櫃買（富櫃50）均線排列", "📈 現貨大聯軍"),
                         ("^SOX", "費城半導體趨勢", "🌐 全商品指揮")):
    try:
      d = fx_add_indicators(fx_fetch(sym, "1y"))
      r = d.iloc[-1]
      tr = _trend(r)
      votes.append(_vote(name, src, tr, f"{TREND_TXT[tr]}（收 {r.Close:,.1f}、MA20 {r.MA20:,.1f}）",
                         f"站回 MA20（{r.MA20:,.1f}）" if tr <= 0 else f"跌破 MA20（{r.MA20:,.1f}）"))
    except Exception as e:  # noqa: BLE001
      votes.append(_vote(name, src, 0, f"資料取得失敗：{e}", "—"))

  try:
    vix = float(fx_fetch("^VIX", "1mo").Close.iloc[-1])
    v = 1 if vix < VIX_CALM else -1 if vix > VIX_FEAR else 0
    vt = "平穩" if v > 0 else "恐慌" if v < 0 else "警戒"
    info["vix"] = vix
    votes.append(_vote("美股恐慌指數 VIX", "⚔️ 選擇權／🌐 全商品", v, f"{vix:.1f}（{vt}）",
                       f"回落到 {VIX_FEAR:.0f} 以下" if v < 0 else f"升破 {VIX_FEAR:.0f}" if v > 0
                       else f"跌破 {VIX_CALM:.0f} 轉多、升破 {VIX_FEAR:.0f} 轉空"))
  except Exception as e:  # noqa: BLE001
    votes.append(_vote("美股恐慌指數 VIX", "⚔️ 選擇權／🌐 全商品", 0, f"資料取得失敗：{e}", "—"))

  try:
    pcr = taifex_pcr()
    oi, vol = float(pcr.oi_pcr.iloc[-1]), float(pcr.vol_pcr.iloc[-1])
    p, pt = pcr_signal(oi)
    info.update(pcr=oi, pcr_date=pcr.index[-1].date().isoformat())
    votes.append(_vote("選擇權未平倉 PCR", "⚔️ 選擇權（期交所）", p,
                       f"{oi:.1f}%（{pt}；5 日均 {pcr.oi_pcr.tail(5).mean():.1f}、20 日均 {pcr.oi_pcr.tail(20).mean():.1f}；"
                       f"成交量 PCR {vol:.1f}%；{info['pcr_date']}）",
                       f"回升到 {PCR_BEAR:.0f} 以上轉中性、{PCR_BULL:.0f} 以上轉多" if p < 0
                       else f"跌破 {PCR_BULL:.0f} 轉中性" if p > 0
                       else f"≥ {PCR_BULL:.0f} 轉多、≤ {PCR_BEAR:.0f} 轉空"))
  except Exception as e:  # noqa: BLE001
    votes.append(_vote("選擇權未平倉 PCR", "⚔️ 選擇權（期交所）", 0, f"資料取得失敗：{e}", "—"))

  return votes, info


def regime_of(total: int) -> str:
  if total >= REGIME_THRESHOLD:
    return "🟢 偏多"
  if total <= -REGIME_THRESHOLD:
    return "🔴 偏空"
  return "🟡 中性"


# ---------- ② 選股掃描 ----------
def _scan_one(code: str, deps: dict) -> dict | None:
  df = deps["fetch"](code, range_str="1y")
  if df is None or df.empty or len(df) < 80:
    return None
  res = deps["confluence"](df)
  c = df["Close"]
  close, prev = float(c.iloc[-1]), float(c.iloc[-2])
  ma13 = float(c.rolling(13).mean().iloc[-1])
  ma60 = c.rolling(60).mean().iloc[-1]
  ma120 = c.rolling(120).mean().iloc[-1]
  atr = float(deps["atr"](df, 14).iloc[-1])
  vol = df["Volume"]
  vol_ratio = float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-8))
  gap_up = float(df["Low"].iloc[-1]) > float(df["High"].iloc[-2])
  gap_dn = float(df["High"].iloc[-1]) < float(df["Low"].iloc[-2])
  hits = [n for n, ok in (("Fibonacci", res["fib"]), ("MACD強化", res["macd"]),
                          ("活躍籌碼", res["mfi_ok"]), ("量能扣底", res["comb"])) if ok]
  return {
    "代號": code, "名稱": deps["names"].get(code, ""), "日期": df.index[-1].date().isoformat(), "收盤": round(close, 2),
    "漲跌%": round((close / prev - 1) * 100, 2), "共振分數": res["score"], "評等": res["grade"],
    "命中戰法": "、".join(hits) or "—", "13MA": round(ma13, 2), "守13MA": close >= ma13,
    "60MA": round(float(ma60), 2) if pd.notna(ma60) else None,
    "120MA": round(float(ma120), 2) if pd.notna(ma120) else None,
    "ATR14": round(atr, 2), "量比": round(vol_ratio, 2),
    "缺口": "⬆️ 跳空向上" if gap_up else "⬇️ 跳空向下" if gap_dn else "",
  }


@st.cache_data(ttl=1800, show_spinner=False)
def scan_group(codes: tuple, _deps: dict) -> pd.DataFrame:
  with ThreadPoolExecutor(max_workers=8) as ex:
    rows = [r for r in ex.map(lambda c: _scan_one(c, _deps), codes) if r]
  return pd.DataFrame(rows)


def radar_chips() -> tuple[dict, str]:
  """雷達艙每日掃描的法人籌碼（最新一天）。取不到就回傳空 dict。"""
  try:
    from radar_cabin import load_json
    scan = load_json("scan.json")
  except Exception:  # noqa: BLE001
    return {}, "雷達資料取得失敗"
  if not scan:
    return {}, "雷達尚無掃描資料"
  df = pd.DataFrame(scan)
  day = df["date"].max()
  out = {}
  for _, r in df[df["date"] == day].iterrows():
    c = r["chips"] if isinstance(r.get("chips"), dict) else {}
    out[str(r["stock_id"])] = {"inst_5d": c.get("inst_5d"), "signal": r.get("signal"), "level": r.get("level")}
  return out, day


def _chip_text(code: str, chips: dict) -> str:
  c = chips.get(code)
  if not c or c.get("inst_5d") is None:
    return "—"
  v = c["inst_5d"]
  return f"✅ 法人買超 {v:,.0f} 張" if v > 0 else f"⚠️ 法人賣超 {abs(v):,.0f} 張" if v < 0 else "法人持平"


# ---------- ④ 風控 ----------
def build_plan(scan: pd.DataFrame, regime: str, capital: float, chips: dict, deps: dict) -> pd.DataFrame:
  cap_pct, risk_pct, min_score = REGIMES[regime]
  if scan.empty:
    return scan
  pick = scan[(scan["共振分數"] >= min_score) & scan["守13MA"]].copy()
  if pick.empty:
    return pick
  pick["籌碼"] = pick["代號"].map(lambda c: _chip_text(c, chips))
  pick = pick[~pick["籌碼"].str.startswith("⚠️")]          # 法人賣超者剔除
  if pick.empty:
    return pick
  pick = pick.sort_values(["共振分數", "量比"], ascending=False)
  mgr = deps["risk_mgr"](total_capital=capital, risk_tolerance_pct=risk_pct)
  pick["停損"] = (pick["收盤"] - ATR_STOP_MULT * pick["ATR14"]).round(2)
  pick["股數"] = [mgr.calculate_trade_risk(c, p, s, 0, 0, 0, False, m)["shares"]
                 for c, p, s, m in zip(pick["代號"], pick["收盤"], pick["停損"], pick["13MA"])]
  pick["金額"] = pick["股數"] * pick["收盤"]
  budget = capital * cap_pct / 100
  if pick["金額"].sum() > budget:                       # 超過總部位上限 → 等比例縮小
    ratio = budget / pick["金額"].sum()
    pick["股數"] = (pick["股數"] * ratio).astype(int)
    pick["金額"] = pick["股數"] * pick["收盤"]
  pick["張數"] = pick["股數"].map(lambda s: f"{s} 股（零股）" if s < 1000
                                else f"{s // 1000} 張 {s % 1000} 股" if s % 1000 else f"{s // 1000} 張")
  return pick


# ---------- 報告 ----------
def _md_table(df: pd.DataFrame) -> str:
  cols = list(df.columns)
  lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
  for _, r in df.iterrows():
    lines.append("| " + " | ".join(str(r[c]).replace("|", "／") for c in cols) + " |")
  return "\n".join(lines)


def _html_table(df: pd.DataFrame) -> str:
  head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
  body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                 for r in df.itertuples(index=False))
  return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_sections(votes, info, total, regime, plan, scan, chips_day, group, capital, n_codes=None,
                   buys=None, sells=None, preopen_fc=None) -> list[tuple[str, list]]:
  """回傳 [(標題, [段落 str 或 DataFrame, ...]), ...]，Markdown 與 HTML 共用"""
  cap_pct, risk_pct, min_score = REGIMES[regime]
  n_up, n_dn = sum(v["票"] > 0 for v in votes), sum(v["票"] < 0 for v in votes)
  n_fail = (n_codes or len(scan)) - len(scan)
  scanned = f"「{group}」共掃描 {len(scan)} 檔" + (f"（另 {n_fail} 檔資料不足或抓取失敗）" if n_fail > 0 else "")
  vt = pd.DataFrame(votes)[["項目", "來源艙", "票", "讀數"]].assign(票=lambda d: d["票"].map(VOTE_ICON))

  sec = [("一、大盤環境", [
    f"大盤燈號：**{regime}**（{len(votes)} 票：{n_up} 多、{n_dn} 空，合計 {total:+d}）。"
    f"加權指數 {info['twii']:,.2f}（{info['twii_chg']:+.2f}%），RSI {info['twii_rsi']:.1f}；"
    f"USD/TWD {info['usdtwd']:.3f}。",
    vt,
  ])]

  if plan is None or plan.empty:
    body = [f"{scanned}，"
            f"在 {regime} 門檻（共振 ≥ {min_score} 分、守住 13MA、法人未賣超）下**沒有入選標的**，建議空手觀望。"]
    near = scan[scan["共振分數"] >= min_score - 1].sort_values("共振分數", ascending=False).head(5) if not scan.empty else scan
    if not near.empty:
      body += ["差一步的觀察名單：", near[["代號", "名稱", "收盤", "共振分數", "命中戰法", "守13MA"]]]
  else:
    body = [f"{scanned}，入選 {len(plan)} 檔（共振 ≥ {min_score} 分、守住 13MA、法人未賣超）。",
            plan[["代號", "名稱", "收盤", "漲跌%", "共振分數", "評等", "命中戰法", "籌碼"]]]
  sec.append(("二、選股名單", body))

  timing = scan[(scan["缺口"] != "") | (scan["量比"] >= 2)] if not scan.empty else scan
  sec.append(("三、進場時機", [
    "今日出現跳空缺口或量比 ≥ 2 倍的標的（量比 = 今日量 ÷ 20 日均量；盤中數字會變動）：",
    timing[["代號", "名稱", "收盤", "漲跌%", "缺口", "量比", "共振分數"]] if not timing.empty else "今日無。",
  ]))

  risk = [f"依燈號 {regime}：總部位上限 **{cap_pct}%**（{capital * cap_pct / 100:,.0f} 元）、"
          f"單筆風險 {risk_pct}% 資金、停損 = 收盤 − {ATR_STOP_MULT:g}×ATR14，並以 13MA 為防線。"]
  if plan is not None and not plan.empty:
    risk += [plan[["代號", "名稱", "收盤", "停損", "13MA", "張數", "金額"]].assign(金額=lambda d: d["金額"].map("{:,.0f}".format)),
             f"合計動用 {plan['金額'].sum():,.0f} 元（{plan['金額'].sum() / capital * 100:.1f}% 資金）。"]
  sec.append(("四、風控", risk))

  trade = []
  if buys is not None and not buys.empty:
    trade += [f"**買進（{len(buys)} 檔）**：隔日開盤 ≤ 開盤上限才買（不追高），或拉回 13MA 掛單；"
              "漲到「停利一半」先賣一半，其餘以 13MA 移動停利。", buys]
  else:
    trade += [f"**買進**：無（{regime} 門檻下沒有入選標的）。"]
  if sells is not None and not sells.empty:
    cnt = sells["訊號"].value_counts()
    trade += [f"**持股檢查（{len(sells)} 檔）**：" + "、".join(f"{k} {v} 檔" for k, v in cnt.items()) + "。",
              sells[["訊號", "代號", "名稱", "類型", "出場規則", "今收", "出場線", "建議賣出股數", "理由"]]]
  else:
    trade += ["**持股檢查**：尚未輸入持股（在上方「💼 我的持股」填寫）。"]
  trade += ["規則：每檔依自己的出場規則，🔴 收盤跌破出場線（均線 ×（1 − 緩衝））→ 賣出；"
            "交易持股另有停損（成本 − 2×ATR14）與獲利 ≥ 2R 賣一半，核心持股不套用、跌破 13MA 只提醒；"
            "🟡「60MA 減半、120MA 出清」跌破 60MA 出場線 → 賣一半；"
            "🟠 總持股超過燈號部位上限 → 等比例減碼；🟢 其餘續抱（出場線為隔日警戒價）。"
            "訊號是規則的機械輸出，未經回測驗證，不構成投資建議。"]
  sec.append(("五、隔日交易訊號", trade))

  watch = [f"- {v['項目']}：{v['翻轉條件']}" for v in votes if v["票"] != 0 and v["翻轉條件"] != "—"]
  five = ["\n".join(watch) if watch else "各項皆為中性，等待方向表態。"]
  lv = info.get("levels")
  if lv is not None and not lv.empty:
    sup = lv[lv["位置"] == "支撐"].head(3)                 # 表格由高到低排序：支撐取前幾列＝離現價最近
    res = lv[lv["位置"] == "壓力"].tail(3).iloc[::-1]      # 壓力取最後幾列再反轉＝由近到遠

    def _fmt(d):
      return "、".join(f"{r['價位']}（{r['類型']}，{r['距現價%']:+.2f}%）" for _, r in d.iterrows()) or "無"
    five += [f"加權指數支撐／壓力（由近到遠）：\n- 下檔支撐：{_fmt(sup)}\n- 上檔壓力：{_fmt(res)}", lv]
  sec.append(("六、明日觀察", five))
  if preopen_fc is not None:
    import preopen
    h, body = preopen.section(preopen_fc, sells=sells, buys=buys)
    sec.append((f"七、{h}", body))

  notes = [f"資料截至 {info['date']}（若在台股盤中產生，當日 K 棒尚未收盤）。"
           f"籌碼來自雷達艙 {chips_day or '（未取得）'} 的掃描，僅涵蓋雷達自選股。",
           "期現價差需要永豐連線，本版未納入投票；回測艙的績效為示範值，未採用。開盤前預測的夜盤來自期交所盤後行情（夜盤進行中改用永豐即時價）。PCR 為期交所每日盤後公布值。",
           "本報告由固定規則產生，僅供技術分析參考，不構成投資建議。"]
  sec.append(("附註", notes))
  return sec


def to_markdown(title: str, sections) -> str:
  out = [f"# {title}"]
  for h, items in sections:
    out.append(f"\n## {h}\n")
    for it in items:
      out.append(_md_table(it) if isinstance(it, pd.DataFrame) else it)
      out.append("")
  return "\n".join(out)


def to_html(title: str, sections) -> str:
  def para(s: str) -> str:
    s = html.escape(s)
    while "**" in s:
      s = s.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return f"<p>{s.replace(chr(10), '<br>')}</p>"
  body = "".join(
    f"<h2>{html.escape(h)}</h2>" + "".join(_html_table(it) if isinstance(it, pd.DataFrame) else para(it) for it in items)
    for h, items in sections)
  css = ("body{font-family:'Noto Sans TC','Microsoft JhengHei',sans-serif;max-width:960px;margin:2em auto;"
         "padding:0 1em;color:#222;line-height:1.6}h1{border-bottom:3px solid #e03131}"
         "table{border-collapse:collapse;width:100%;margin:.5em 0;font-size:14px}"
         "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}th{background:#f3f3f3}")
  return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
          f"<style>{css}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>")


# ---------- 關鍵指標線圖（費半、VIX） ----------
@st.cache_data(ttl=1800, show_spinner="下載費半與 VIX 線圖資料…")
def chart_data(symbol: str) -> pd.DataFrame:
  # 多抓一些再丟掉前 40 根：RSI／MACD 剛起算時數值不穩（RSI 會出現 100 或 0）
  df = fx_add_indicators(fx_fetch(symbol, "2y")).iloc[40:]
  return df[df.index > df.index[-1] - pd.Timedelta(days=365)]


def sox_figure(df: pd.DataFrame) -> go.Figure:
  fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.22, 0.18],
                      subplot_titles=("費城半導體指數 ^SOX（K 線＋MA20／MA60＋布林）", "MACD (12, 26, 9)", "RSI (14)"))
  fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="SOX",
                               increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
  for col, color, dash in (("MA20", "#f08c00", None), ("MA60", "#1c7ed6", None),
                           ("BB_up", "#adb5bd", "dot"), ("BB_dn", "#adb5bd", "dot")):
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=1.2, dash=dash)), 1, 1)
  ma20 = float(df.MA20.iloc[-1])
  fig.add_hline(y=ma20, line_dash="dash", line_color="#f08c00", row=1, col=1,
                annotation_text=f"翻轉線 MA20 {ma20:,.1f}", annotation_position="top left")
  _fx_macd_panel(fig, df, 2)
  fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", showlegend=False,
                           line=dict(color="#7048e8", width=1.3)), 3, 1)
  for lvl in (70, 30):
    fig.add_hline(y=lvl, line_dash="dash", line_color="#adb5bd", row=3, col=1)
  return _fx_layout(fig, 780)


def vix_figure(df: pd.DataFrame) -> go.Figure:
  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.72, 0.28],
                      subplot_titles=("美股恐慌指數 ^VIX（上漲＝恐慌升高）", "RSI (14)"))
  top = max(float(df.High.max()) * 1.05, VIX_FEAR + 5)
  bottom = max(0.0, min(float(df.Low.min()) * 0.95, VIX_CALM - 4))
  for y0, y1, color, label in ((bottom, VIX_CALM, "rgba(224,49,49,0.22)", "平穩（多方 +1）"),
                               (VIX_CALM, VIX_FEAR, "rgba(173,181,189,0.12)", "警戒（0）"),
                               (VIX_FEAR, top, "rgba(47,158,68,0.25)", "恐慌（空方 −1）")):
    fig.add_shape(type="rect", xref="x domain", x0=0, x1=1, yref="y", y0=y0, y1=y1,
                  fillcolor=color, line_width=0, layer="below")
    fig.add_annotation(xref="x domain", x=0.005, yref="y", y=(y0 + y1) / 2, text=label, showarrow=False,
                       xanchor="left", font=dict(size=12, color="#dee2e6"))
  fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="VIX",
                               increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name="MA20", line=dict(color="#f08c00", width=1.2)), 1, 1)
  for lvl, txt in ((VIX_CALM, f"{VIX_CALM:.0f}"), (VIX_FEAR, f"{VIX_FEAR:.0f} 恐慌線")):
    fig.add_hline(y=lvl, line_dash="dash", line_color="#ced4da", row=1, col=1,
                  annotation_text=txt, annotation_position="bottom right")
  fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", showlegend=False,
                           line=dict(color="#7048e8", width=1.3)), 2, 1)
  for lvl in (70, 30):
    fig.add_hline(y=lvl, line_dash="dash", line_color="#adb5bd", row=2, col=1)
  fig.update_yaxes(range=[bottom, top], row=1, col=1)
  return _fx_layout(fig, 640)


def pcr_figure(df: pd.DataFrame) -> go.Figure:
  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38],
                      subplot_titles=("臺指選擇權 Put/Call Ratio（%）", "未平倉量（口）：賣權 vs 買權"))
  top = max(float(df.oi_pcr.max()), float(df.vol_pcr.max())) * 1.08
  bottom = min(float(df.oi_pcr.min()), float(df.vol_pcr.min()), PCR_BEAR) * 0.9
  for y0, y1, color, label in ((PCR_BULL, max(top, PCR_BULL + 10), "rgba(224,49,49,0.20)", "支撐強・偏多（+1）"),
                               (PCR_BEAR, PCR_BULL, "rgba(173,181,189,0.10)", "中性（0）"),
                               (bottom, PCR_BEAR, "rgba(47,158,68,0.22)", "支撐弱・偏空（−1）")):
    fig.add_shape(type="rect", xref="x domain", x0=0, x1=1, yref="y", y0=y0, y1=y1,
                  fillcolor=color, line_width=0, layer="below")
    fig.add_annotation(xref="x domain", x=0.005, yref="y", y=(y0 + y1) / 2, text=label, showarrow=False,
                       xanchor="left", font=dict(size=12, color="#dee2e6"))
  fig.add_trace(go.Scatter(x=df.index, y=df.oi_pcr, name="未平倉 PCR（投票用）", mode="lines+markers",
                           line=dict(color="#ffd43b", width=2), marker=dict(size=4)), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.oi_pcr.rolling(20).mean(), name="未平倉 PCR 20 日均",
                           line=dict(color="#f08c00", width=1.2, dash="dot")), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.vol_pcr, name="成交量 PCR", line=dict(color="#74c0fc", width=1)), 1, 1)
  for lvl in (PCR_BULL, PCR_BEAR):
    fig.add_hline(y=lvl, line_dash="dash", line_color="#ced4da", row=1, col=1,
                  annotation_text=f"{lvl:.0f}", annotation_position="bottom right")
  fig.add_trace(go.Bar(x=df.index, y=df.put_oi, name="賣權未平倉", marker_color=UP), 2, 1)
  fig.add_trace(go.Bar(x=df.index, y=df.call_oi, name="買權未平倉", marker_color=DN), 2, 1)
  fig.update_layout(barmode="group")
  fig.update_yaxes(range=[bottom, max(top, PCR_BULL + 10)], row=1, col=1)
  return _fx_layout(fig, 640)


def render_key_charts():
  st.markdown("#### 📈 關鍵指標線圖")
  t1, t2, t3 = st.tabs(["🔬 費城半導體 SOX", "😱 美股恐慌指數 VIX", "⚖️ 台指選擇權 PCR"])
  with t1:
    try:
      df = chart_data("^SOX")
      r = df.iloc[-1]
      gap = (r.Close / r.MA20 - 1) * 100
      st.caption(f"收 {r.Close:,.1f}／MA20 {r.MA20:,.1f}（距離 {gap:+.2f}%）／MA60 {r.MA60:,.1f}／RSI {r.RSI:.1f}。"
                 "翻轉條件：收盤站回 MA20 且 MA20 在 MA60 之上才是多頭排列。")
      st.plotly_chart(sox_figure(df), key="sm_sox_chart")
    except Exception as e:  # noqa: BLE001
      st.warning(f"費半資料取得失敗：{e}")
  with t2:
    try:
      df = chart_data("^VIX")
      r = df.iloc[-1]
      st.caption(f"VIX {r.Close:.1f}（MA20 {r.MA20:.1f}）。規則：< {VIX_CALM:.0f} 平穩 +1、"
                 f"{VIX_CALM:.0f}～{VIX_FEAR:.0f} 警戒 0、> {VIX_FEAR:.0f} 恐慌 −1；翻轉條件：回落到 {VIX_FEAR:.0f} 以下。")
      st.plotly_chart(vix_figure(df), key="sm_vix_chart")
    except Exception as e:  # noqa: BLE001
      st.warning(f"VIX 資料取得失敗：{e}")
  with t3:
    try:
      df = taifex_pcr()
      r = df.iloc[-1]
      sig, txt = pcr_signal(r.oi_pcr)
      st.caption(f"{df.index[-1].date()} 未平倉 PCR {r.oi_pcr:.1f}%（{txt}）、成交量 PCR {r.vol_pcr:.1f}%；"
                 f"賣權未平倉 {r.put_oi:,.0f} 口、買權未平倉 {r.call_oi:,.0f} 口。"
                 f"規則：≥ {PCR_BULL:.0f} 偏多 +1、≤ {PCR_BEAR:.0f} 偏空 −1。"
                 "看法：賣 Put 的多半是法人，未平倉 PCR 高代表下方支撐強。資料為期交所盤後公布值（約 15:00 後更新）。")
      st.plotly_chart(pcr_figure(df), key="sm_pcr_chart")
    except Exception as e:  # noqa: BLE001
      st.warning(f"期交所 PCR 取得失敗：{e}")


# ---------- 介面 ----------
def _preopen_forecast(deps):
  """開盤前 6 大指標；失敗回傳 None（不影響總結報告其他段落）"""
  import preopen
  api = deps["sj"]() if "sj" in deps else None
  try:
    return preopen.forecast(api, deps.get("resolve"), api is not None)
  except Exception:  # noqa: BLE001
    return None


def render_summary_cabin(deps: dict):
  """deps：fetch, confluence, atr, risk_mgr, groups, names（由 app.py 傳入）"""
  import twse_data
  deps = {**deps, "fetch": twse_data.fresh_fetch(deps["fetch"])}   # Yahoo 日 K 落後時用官方資料補齊
  st.subheader("📋 每日綜合戰情總結艙（大盤投票 × 多策略共振 × 籌碼確認 × 風控 → 一頁總結報告）")
  st.caption("全部由固定規則產生、不接 AI：每個結論都能追到是哪一艙、哪個數字。僅供參考，不構成投資建議。")

  hm = pd.Timestamp.now(tz="Asia/Taipei")
  morning = (7, 0) <= (hm.hour, hm.minute) <= (9, 30)        # 早上 07:00～09:30 自動展開
  with st.expander("🌅 開盤前趨勢預測（08:45 台指期／09:00 台股；ADR、費半、道瓊、匯率、KOSPI、夜盤）", expanded=morning):
    st.caption("建議 08:00～08:45 之間看：此時美股與夜盤已收盤、KOSPI 已開盤。也會自動併入總結報告第七段。")
    if st.button("🌅 抓取開盤前 6 大指標", key="sm_pre_run"):
      st.session_state["sm_pre"] = _preopen_forecast(deps)
    fc = st.session_state.get("sm_pre")
    if fc is not None:
      import preopen
      trade = st.session_state.get("sm_trade") or (None, None, 0)
      h, body = preopen.section(fc, sells=trade[1], buys=trade[0])
      for it in body:
        if isinstance(it, pd.DataFrame):
          st.dataframe(it, hide_index=True, width="stretch")
        else:
          st.markdown(it)
    elif "sm_pre" in st.session_state:
      st.warning("開盤前指標抓取失敗（查不到台股最新交易日或網路問題），請稍後再按一次。")

  groups = deps["groups"]
  keys = list(groups.keys())
  with st.expander("⚙️ 本艙設定", expanded=False):
    c1, c2 = st.columns([2, 1])
    group = c1.selectbox("選股掃描族群", keys,
                         index=keys.index(DEFAULT_GROUP) if DEFAULT_GROUP in keys else 0, key="sm_group")
    capital = c2.number_input("帳戶總資金（元）", value=1_000_000.0, step=100_000.0, min_value=10_000.0, key="sm_cap")
    st.caption("燈號規則：8 票合計 ≥ +3 為 🟢 偏多、≤ −3 為 🔴 偏空，其餘 🟡 中性。"
               "🟢 部位上限 80%／單筆風險 1%／共振 ≥ 3；🟡 50%／0.75%／≥ 4；🔴 20%／0.5%／≥ 4。")
    if len(groups[group]) > 60:
      st.warning(f"此族群有 {len(groups[group])} 檔，掃描約需 1～3 分鐘。")

  _NAMES.update(deps.get("names") or {})
  holdings = er.effective(render_holdings_editor())
  if "sj" in deps:
    with st.expander("⏱️ 盤中監控（持股賣出／續抱、候選買點；訊號改變才提醒）", expanded=False):
      import intraday_monitor as im
      trade = st.session_state.get("sm_trade")
      cands = list(trade[0]["代號"]) if trade and trade[0] is not None and not trade[0].empty else []
      im.render_monitor(deps, holdings, cands, deps["names"])

  b1, b2 = st.columns([1, 1])
  run = b1.button("🚀 產生今日總結報告", type="primary", key="sm_run")
  if b2.button("🔄 清除快取重新抓取", key="sm_refresh"):
    market_votes.clear()
    scan_group.clear()
    chart_data.clear()
    taifex_pcr.clear()
    fx_build_link.clear()
    import twse_data
    twse_data.clear_cache()
    import preopen
    preopen.forecast.clear()
    preopen.night_session.clear()
    if "sj" in deps:
      import intraday_monitor as im
      im.daily_hist.clear()
    st.session_state.pop("sm_report", None)
    st.toast("快取已清除，請按「🚀 產生今日總結報告」重新抓取。")

  if run:
    try:
      votes, info = market_votes()
    except Exception as e:  # noqa: BLE001
      st.error(f"大盤資料取得失敗：{e}")
      return
    with st.spinner(f"掃描「{group}」{len(groups[group])} 檔…"):
      scan = scan_group(tuple(groups[group]), deps)
    chips, chips_day = radar_chips()
    total = sum(v["票"] for v in votes)
    regime = regime_of(total)
    plan = build_plan(scan, regime, capital, chips, deps)
    # 隔日交易訊號：買進來自入選名單；賣出逐檔檢查持股
    hold_scan = scan_group(tuple(holdings["代號"]), deps) if not holdings.empty else pd.DataFrame()
    buys = ts.buy_signals(plan, set(holdings["代號"]))
    sells = ts.sell_signals(holdings, hold_scan, REGIMES[regime][0], capital)
    added = ts.append_log(info["date"], regime, buys, sells)
    fc = _preopen_forecast(deps)
    title = f"每日綜合戰情總結報告 {info['date']}"
    st.session_state["sm_report"] = (title, build_sections(votes, info, total, regime, plan, scan,
                                                           chips_day, group, capital, len(groups[group]),
                                                           buys=buys, sells=sells, preopen_fc=fc),
                                  regime, total, info, len(votes))
    st.session_state["sm_trade"] = (buys, sells, added)
    st.session_state["sm_report_at"] = pd.Timestamp.now(tz="Asia/Taipei").strftime("%m/%d %H:%M")

  rep = st.session_state.get("sm_report")
  if not rep:
    st.info("按「🚀 產生今日總結報告」開始。大盤與掃描結果快取 30 分鐘。")
    return
  title, sections, regime, total, info, n_votes = rep

  k = st.columns(4)
  k[0].metric("大盤燈號", regime, f"{total:+d} / {n_votes} 票", delta_color="off")
  k[1].metric("加權指數", f"{info['twii']:,.2f}", f"{info['twii_chg']:+.2f}%", delta_color="inverse")
  k[2].metric("股匯相關（60 日）", f"{info['corr']:+.2f}")
  k[3].metric("VIX", f"{info['vix']:.1f}" if "vix" in info else "—")

  st.markdown(f"### {title}")
  at = st.session_state.get("sm_report_at")
  if at:
    st.caption(f"報告產生時間 {at}（資料日期 {info['date']}）。報告不會自己更新，要看最新數據請重按「🚀 產生今日總結報告」；"
               "大盤與掃描結果快取 30 分鐘，要立刻重抓請先按「🔄 清除快取重新抓取」。")
  for h, items in sections:
    st.markdown(f"#### {h}")
    for it in items:
      if isinstance(it, pd.DataFrame):
        st.dataframe(it, hide_index=True, width="stretch")
      else:
        st.markdown(it)

  trade = st.session_state.get("sm_trade")
  if trade and trade[2]:
    st.caption(f"📒 已把今天 {trade[2]} 筆新訊號記入戰績。")

  ai_section = render_ai_section(title, sections)
  render_key_charts()
  render_track_record(deps)

  out = sections + ([ai_section] if ai_section else [])
  d1, d2 = st.columns(2)
  d1.download_button("⬇️ 下載 Markdown", to_markdown(title, out).encode("utf-8"),
                     file_name=f"{title}.md", mime="text/markdown", key="sm_dl_md")
  d2.download_button("⬇️ 下載 HTML", to_html(title, out).encode("utf-8"),
                     file_name=f"{title}.html", mime="text/html", key="sm_dl_html")


def render_ai_section(title: str, sections) -> tuple[str, list] | None:
  """八、AI 研判：按鈕觸發才呼叫 Claude。回傳要併進下載檔的段落（沒產生就回傳 None）。"""
  import ai_brief

  st.markdown("#### 八、🧠 AI 研判（Claude 撰寫，僅供參考）")
  key, why = ai_brief.key_status()
  if not key:
    st.warning(f"讀不到金鑰：{why}")
    st.info("尚未設定 Anthropic API 金鑰：本機寫在 `.streamlit/secrets.toml` 的 `ANTHROPIC_API_KEY = \"sk-ant-...\"`，"
            "Streamlit Cloud 在 App settings → Secrets 加同一行。設定後重新整理即可使用。")
    return None
  report_md = to_markdown(title, sections)
  model = ai_brief.model_name()
  state_key = f"sm_ai::{hash(report_md)}::{model}"
  if st.button("🧠 產生 AI 研判", key="sm_ai_run",
               help=f"把上面的規則報告交給 {model} 寫成研判段落。燈號、名單、部位、停損不會被改動。同一份報告 24 小時內只計費一次。"):
    with st.spinner(f"{model} 撰寫中（約 10～60 秒）…"):
      try:
        st.session_state[state_key] = ai_brief.generate(report_md, model, key)
      except Exception as e:  # noqa: BLE001
        st.error(f"AI 研判產生失敗：{ai_brief.error_text(e)}")
  res = st.session_state.get(state_key)
  if not res:
    st.caption("按下按鈕才會呼叫 AI（會產生 API 費用）。規則報告本身不受影響。")
    return None
  st.markdown(res["text"])
  bad = ai_brief.unmatched_numbers(res["text"], report_md)
  if bad:
    st.warning("⚠️ 以下數字在規則報告裡找不到，可能是 AI 自行推算或寫錯，請以上方規則報告為準：" + "、".join(bad))
  else:
    st.caption("✅ 數字核對：AI 文中的數字都能在規則報告裡找到。")
  st.caption(f"模型 {res['model']}；輸入 {res['in_tok']:,} tokens、輸出 {res['out_tok']:,} tokens。")
  note = ("⚠️ 數字核對未通過：" + "、".join(bad)) if bad else "數字核對：通過。"
  return ("八、AI 研判（Claude 撰寫，僅供參考）", [res["text"], f"{note}（模型 {res['model']}）"])


_NAMES: dict = {}


def render_holdings_editor() -> pd.DataFrame:
  """💼 我的持股：給賣出訊號用。只存在本機 data/holdings.csv（不進 GitHub）；雲端請用下載／上傳 CSV。"""
  if "sm_hold" not in st.session_state:
    st.session_state["sm_hold"] = ts.load_holdings()
  with st.expander(f"💼 我的持股（{len(st.session_state['sm_hold'])} 檔，賣出訊號用）", expanded=False):
    st.caption("填代號即可產生賣出訊號；填成本才會檢查停損與 2R 停利（只用在交易持股）；填股數才會計算總部位與減碼股數。"
               "「類型」留空＝依股性自動建議，「出場規則」留空＝依類型預設（核心 60MA 加 2%、交易 13MA 加 1%）。"
               + ("持股存在本機 data/holdings.csv（不會上傳到 GitHub）。" if ts.files_enabled()
                  else "持股只存在這個瀏覽器分頁，其他訪客看不到；請下載 CSV 保存，下次再上傳。"))
    up = st.file_uploader("上傳持股 CSV（欄位：代號, 成本, 股數，可另加 類型, 出場規則）", type="csv", key="sm_hold_up")
    if up is not None and st.session_state.get("sm_hold_up_name") != up.name:
      try:
        st.session_state["sm_hold"] = ts._clean_holdings(pd.read_csv(up, dtype={"代號": str}, nrows=500))
      except Exception:  # noqa: BLE001
        st.warning("無法讀取這個 CSV，請確認欄位是 代號, 成本, 股數。")
      st.session_state["sm_hold_up_name"] = up.name
    edited = st.data_editor(
      st.session_state["sm_hold"], num_rows="dynamic", hide_index=True, key="sm_hold_editor",
      column_config={
        "代號": st.column_config.TextColumn("代號", help="台股代號，例如 2330、0050"),
        "成本": st.column_config.NumberColumn("成本（元／股）", min_value=0.0, format="%.2f"),
        "股數": st.column_config.NumberColumn("股數", min_value=0, step=1, format="%d", help="1 張 = 1000 股"),
        "類型": st.column_config.SelectboxColumn("類型", options=er.TYPES,
                                                help="核心：長抱、不套用短線停損；交易：短線規則。留空＝依股性自動建議"),
        "出場規則": st.column_config.SelectboxColumn("出場規則", options=list(er.RULES), width="medium",
                                                  help="留空＝核心用 60MA 加 2% 緩衝、交易用 13MA 加 1% 緩衝"),
      })
    hold = ts._clean_holdings(edited)
    if not hold.empty:
      eff = er.effective(hold)
      prof = pd.DataFrame([er.profile(c) for c in hold["代號"]])
      view = eff[["代號", "類型", "出場規則", "設定來源"]].merge(prof, on="代號", how="left")
      view.insert(1, "名稱", view["代號"].map(lambda c: _NAMES.get(c, "")))
      st.markdown("**生效設定與股性**（2009 年以來含息日線；四項都通過才建議「核心」："
                  f"年化波動 ≤ {er.TH_VOL:g}%、年線上方 ≥ {er.TH_ABOVE:g}%、最長套牢 ≤ {er.TH_UNDER:g} 年、"
                  f"單年跌逾 30% ≤ {er.TH_BADYEARS} 次）")
      st.dataframe(view, hide_index=True, width="stretch")
      st.caption("要改就在上表的「類型」「出場規則」欄選擇，再按儲存。股性是歷史統計，不保證未來，不構成投資建議。")
    c1, c2 = st.columns(2)
    if c1.button("💾 儲存持股", key="sm_hold_save"):
      st.session_state["sm_hold"] = hold
      if ts.save_holdings(hold):
        st.success("已儲存到本機 data/holdings.csv。")
      else:
        st.info("已存在這個瀏覽器分頁（其他訪客看不到）；關掉分頁就會消失，請用右邊的下載保存。")
    c2.download_button("⬇️ 下載持股 CSV", hold.to_csv(index=False).encode("utf-8-sig"),
                       file_name="holdings.csv", mime="text/csv", key="sm_hold_dl")
  return hold


def render_track_record(deps: dict):
  """📒 訊號戰績：回頭計算過去訊號 1／5／20 日後的表現。"""
  with st.expander("📒 訊號戰績（每次產生報告自動記錄）", expanded=False):
    up = st.file_uploader("上傳先前下載的戰績紀錄 CSV（與目前紀錄合併）", type="csv", key="sm_log_up")
    if up is not None and st.session_state.get("sm_log_up_name") != up.name:
      try:
        n = ts.merge_upload(pd.read_csv(up, dtype={"代號": str}, nrows=ts.MAX_UPLOAD_ROWS))
        st.success(f"已合併 {n} 筆新紀錄（格式不符的列已略過）。")
      except Exception:  # noqa: BLE001
        st.warning("無法讀取這個 CSV，請確認是本艙下載的戰績紀錄。")
      st.session_state["sm_log_up_name"] = up.name
    log = ts.load_log()
    if log.empty:
      st.caption("還沒有紀錄。之後每次按「🚀 產生今日總結報告」，買進／賣出／賣一半／減碼訊號都會自動記下來。")
      return
    st.caption(f"共 {len(log)} 筆，最早 {log['日期'].min()}。" + ("存在本機 data/signal_log.csv。" if ts.files_enabled() else "只存在這個瀏覽器分頁，請定期下載保存。"))
    if st.button("📊 計算戰績", key="sm_track_run"):
      with st.spinner("抓取後續股價…"):
        st.session_state["sm_track"] = ts.track_record(log, deps["fetch"])
    tr = st.session_state.get("sm_track")
    if tr:
      summary, detail = tr
      if not summary.empty:
        st.markdown("**彙總**：" + summary.attrs.get("note", ""))
        st.dataframe(summary, hide_index=True, width="stretch")
        st.markdown("**明細**（訊號日之後的交易日數還不夠的欄位會空白）")
        st.dataframe(detail, hide_index=True, width="stretch")
    st.download_button("⬇️ 下載戰績紀錄 CSV", log.to_csv(index=False).encode("utf-8-sig"),
                       file_name="signal_log.csv", mime="text/csv", key="sm_log_dl")
