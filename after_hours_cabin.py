"""第 27 艙「🕑 盤後定價艙」。

盤後定價交易：14:00～14:30 委託，14:30 以「當日收盤價」一次撮合，買賣對不上時電腦隨機分配。
  「未成交買量 > 未成交賣量」＝收盤價買不到（買盤較積極）；反之＝收盤價賣不掉（賣壓未消化）。

資料直接抓交易所公開資料（只讀、不需金鑰）：
  上市：證交所 BFT41U（盤後定價，可指定日期）＋ MI_INDEX（盤中成交股數）
  上櫃：櫃買 OpenAPI tpex_off_market（只提供最新一個交易日）＋ afterTrading/otc（盤中成交股數）
單位：盤後成交、未成交量是「張」；交易所的盤中成交是「股」，這裡換成張。
證交所約每 5 秒 3 次請求就會暫時封鎖 IP，所以請求序列化、間隔 0.6 秒，並以交易日為單位快取。
網址加 ?cabin=after 直接打開。
"""

from __future__ import annotations

import datetime as dt
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

UP, DN = "#e03131", "#2f9e44"   # 台灣習慣：紅漲綠跌
TWSE = "https://www.twse.com.tw/rwd/zh/afterTrading"
TPEX_FIXED = "https://www.tpex.org.tw/openapi/v1/tpex_off_market"
TPEX_QUOTE = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"
COLS = ["日期", "代號", "名稱", "市場", "盤後成交(張)", "盤中成交(張)", "盤後占盤中%",
        "收盤價", "未成交買量(張)", "未成交賣量(張)"]
FMT = {
  "盤後成交(張)": st.column_config.NumberColumn(format="localized"),
  "盤中成交(張)": st.column_config.NumberColumn(format="localized"),
  "盤後占盤中%": st.column_config.NumberColumn(format="%.2f"),
  "收盤價": st.column_config.NumberColumn(format="%.2f"),
  "未成交買量(張)": st.column_config.NumberColumn(format="localized"),
  "未成交賣量(張)": st.column_config.NumberColumn(format="localized"),
  "買−賣(張)": st.column_config.NumberColumn(format="localized"),
}

_UA = {"User-Agent": "Mozilla/5.0"}
_lock = threading.Lock()
_last_call = [0.0]


class _Empty(Exception):
  """查無資料（尚未公布／被限流）：丟例外讓快取不要記住空結果"""


def _today() -> dt.date:
  return pd.Timestamp.now(tz="Asia/Taipei").date()   # 雲端主機是 UTC


def _get(url: str):
  with _lock:
    wait = 0.6 - (time.time() - _last_call[0])
    if wait > 0:
      time.sleep(wait)
    try:
      return requests.get(url, headers=_UA, timeout=20).json()
    finally:
      _last_call[0] = time.time()


def _num(s) -> float:
  try:
    return float(str(s).replace(",", "").strip() or 0)
  except ValueError:
    return 0.0


def _finish(rows: list[dict]) -> pd.DataFrame:
  df = pd.DataFrame(rows, columns=COLS)
  if not df.empty:
    g = pd.to_numeric(df["盤中成交(張)"], errors="coerce")
    df["盤後占盤中%"] = (df["盤後成交(張)"] / g.where(g > 0) * 100).round(2)
  return df


# ---------- 純解析（可用假資料測試） ----------
def parse_twse(date: dt.date, fixed: dict | None, quote: dict | None) -> pd.DataFrame:
  if not fixed or fixed.get("stat") != "OK" or not fixed.get("fields"):
    return _finish([])
  reg: dict[str, int] = {}
  if quote and quote.get("stat") == "OK":
    for t in quote.get("tables", []):
      f = t.get("fields") or []
      if "證券代號" in f and "成交股數" in f:
        c, v = f.index("證券代號"), f.index("成交股數")
        reg = {r[c]: round(_num(r[v]) / 1000) for r in t.get("data", [])}
        break
  f = fixed["fields"]
  c, n, v, p, b, a = (f.index(k) for k in ["證券代號", "證券名稱", "成交數量", "成交價", "最後揭示買量", "最後揭示賣量"])
  return _finish([{"日期": date, "代號": r[c], "名稱": r[n].strip(), "市場": "上市",
                   "盤後成交(張)": _num(r[v]), "盤中成交(張)": reg.get(r[c]), "收盤價": _num(r[p]),
                   "未成交買量(張)": _num(r[b]), "未成交賣量(張)": _num(r[a])} for r in fixed.get("data", [])])


def roc_to_date(s: str) -> dt.date:
  s = str(s).strip()            # '1151125' → 2026-11-25
  return dt.date(int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:]))


def parse_tpex(fixed: list | None, quote: dict | None) -> pd.DataFrame:
  if not isinstance(fixed, list) or not fixed:
    return _finish([])
  date = roc_to_date(fixed[0]["Date"])
  reg: dict[str, int] = {}
  t = ((quote or {}).get("tables") or [None])[0]
  if t and t.get("fields"):
    f = [str(x).strip() for x in t["fields"]]
    if "代號" in f and "成交股數" in f:
      c, v = f.index("代號"), f.index("成交股數")
      reg = {r[c]: round(_num(r[v]) / 1000) for r in t.get("data", [])}
  rows = []
  for r in fixed:
    code = str(r.get("SecuritiesCompanyCode", "")).strip()
    rows.append({"日期": date, "代號": code, "名稱": str(r.get("CompanyName", "")).strip(), "市場": "上櫃",
                 "盤後成交(張)": _num(r.get("TradeVolume")), "盤中成交(張)": reg.get(code),
                 "收盤價": _num(r.get("Close")), "未成交買量(張)": _num(r.get("BidVolumeUnexecute")),
                 "未成交賣量(張)": _num(r.get("OfferVolumeUnexecute"))})
  return _finish(rows)


def is_common_stock(code: str) -> bool:
  """4 碼數字、非 00 開頭＝普通股（排除 ETF、權證、特別股）"""
  return len(code) == 4 and code.isdigit() and not code.startswith("00")


def recent_weekdays(today: dt.date, n: int) -> list[dt.date]:
  out, d = [], today
  while len(out) < n:
    if d.weekday() < 5:
      out.append(d)
    d -= dt.timedelta(days=1)
  return out


# ---------- 抓取（快取） ----------
@st.cache_data(ttl=86400, show_spinner=False)
def _twse_day_cached(day: dt.date) -> pd.DataFrame:
  ymd = day.strftime("%Y%m%d")
  fixed = _get(f"{TWSE}/BFT41U?date={ymd}&selectType=ALL&response=json")
  if fixed.get("stat") != "OK":
    if day >= _today():
      raise _Empty("今天的盤後定價尚未公布（14:30 撮合後才有）")
    return _finish([])                       # 過去的日子查無資料＝休市，可以快取
  quote = _get(f"{TWSE}/MI_INDEX?date={ymd}&type=ALLBUT0999&response=json")
  return parse_twse(day, fixed, quote)


def twse_day(day: dt.date) -> pd.DataFrame:
  try:
    return _twse_day_cached(day)
  except Exception:  # noqa: BLE001 — 未公布、限流、連線失敗都當作沒有資料
    return _finish([])


@st.cache_data(ttl=900, show_spinner=False)
def _tpex_latest_cached() -> pd.DataFrame:
  fixed = _get(TPEX_FIXED)
  if not isinstance(fixed, list) or not fixed:
    raise _Empty("櫃買盤後定價查無資料")
  d = roc_to_date(fixed[0]["Date"])
  quote = _get(f"{TPEX_QUOTE}?date={d.year}%2F{d.month:02d}%2F{d.day:02d}&type=EW&response=json")
  return parse_tpex(fixed, quote)


def tpex_latest() -> pd.DataFrame:
  try:
    return _tpex_latest_cached()
  except Exception:  # noqa: BLE001
    return _finish([])


def twse_latest() -> pd.DataFrame:
  for d in recent_weekdays(_today(), 8):
    df = twse_day(d)
    if not df.empty:
      return df
  return _finish([])


def twse_history(code: str, n_days: int) -> pd.DataFrame:
  rows = []
  days = recent_weekdays(_today(), n_days + 8)
  bar = st.progress(0.0, text="下載證交所盤後定價…")
  for i, d in enumerate(days):
    bar.progress((i + 1) / len(days), text=f"下載證交所盤後定價 {d}…")
    df = twse_day(d)
    hit = df[df["代號"] == code]
    if not hit.empty:
      rows.append(hit.iloc[0])
    if len(rows) >= n_days:
      break
  bar.empty()
  return pd.DataFrame(rows, columns=COLS).reset_index(drop=True)


def clear_cache() -> None:
  _twse_day_cached.clear()
  _tpex_latest_cached.clear()


# ---------- 畫面 ----------
def _signal(r) -> str:
  b, a = r["未成交買量(張)"], r["未成交賣量(張)"]
  return "🔴 收盤價買不到" if b > a else ("🟢 收盤價賣不掉" if a > b else "—")


def _with_signal(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["買−賣(張)"] = df["未成交買量(張)"] - df["未成交賣量(張)"]
  df["訊號"] = df.apply(_signal, axis=1) if not df.empty else []
  return df


def _history_chart(h: pd.DataFrame) -> go.Figure:
  h = h.sort_values("日期")
  x = h["日期"].astype(str)
  net = h["未成交買量(張)"] - h["未成交賣量(張)"]
  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.55, 0.45],
                      subplot_titles=("盤後成交（張）與占盤中比例", "未成交買量 − 未成交賣量（張）"),
                      specs=[[{"secondary_y": True}], [{}]])
  fig.add_bar(x=x, y=h["盤後成交(張)"], name="盤後成交", marker_color="#f59f00", row=1, col=1)
  fig.add_scatter(x=x, y=h["盤後占盤中%"], name="占盤中%", mode="lines+markers",
                  line=dict(color="#1c7ed6"), row=1, col=1, secondary_y=True)
  fig.add_bar(x=x, y=net, name="未成交買−賣", row=2, col=1, marker_color=[UP if v > 0 else DN for v in net])
  fig.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=40, b=10), bargap=0.3)
  fig.update_xaxes(type="category")
  fig.update_yaxes(title_text="%", row=1, col=1, secondary_y=True, showgrid=False)
  return fig


def render_after_hours_cabin(groups: dict, names: dict) -> None:
  st.header("🕑 盤後定價艙")
  st.markdown("**盤後定價交易**：14:00～14:30 委託，14:30 以**當日收盤價**一次撮合，買賣對不上時電腦隨機分配。"
              "「未成交買量」大於「未成交賣量」代表有人想用收盤價買卻沒買到（🔴 買盤較積極）；反之是賣壓沒消化（🟢）。")
  c1, c2 = st.columns([4, 1])
  with c2:
    if st.button("🔄 重抓", help="清除盤後定價快取；今天 14:30 後資料剛公布時使用"):
      clear_cache()
      st.rerun()

  with st.spinner("下載證交所、櫃買中心盤後定價…"):
    twse, tpex = twse_latest(), tpex_latest()
  parts = [m for m in (twse, tpex) if not m.empty]
  with c1:
    if parts:
      st.caption("最新資料日：" + "、".join(f"{m['市場'].iloc[0]} {m['日期'].iloc[0]}" for m in parts)
                 + "　（今天的資料要 14:30 撮合後才會公布）")
  if not parts:
    st.warning("暫時抓不到盤後定價資料。可能是證交所限制連線，請等 1～2 分鐘後按「🔄 重抓」。"
               "Streamlit Cloud 主機在美國，若一直抓不到，代表交易所擋了海外連線。")
    return
  mk = _with_signal(pd.concat(parts, ignore_index=True))

  tab_grp, tab_mkt, tab_one = st.tabs(["🎯 戰略分組", "🏆 全市場排行", "🔍 個股歷史"])

  with tab_grp:
    gnames = list(groups)
    default = next((i for i, g in enumerate(gnames) if "半導體" in g), 0)
    gsel = st.selectbox("戰略分組", gnames, index=default, key="ah_group")
    codes = [str(c) for c in groups[gsel]]
    g = mk[mk["代號"].isin(codes)].copy()
    g["名稱"] = g["代號"].map(lambda c: names.get(c)).fillna(g["名稱"])
    missing = [c for c in codes if c not in set(g["代號"])]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("有資料", f"{len(g)} / {len(codes)} 檔")
    m2.metric("盤後成交合計", f"{g['盤後成交(張)'].sum():,.0f} 張")
    m3.metric("🔴 收盤價買不到", f"{(g['買−賣(張)'] > 0).sum()} 檔")
    m4.metric("🟢 收盤價賣不掉", f"{(g['買−賣(張)'] < 0).sum()} 檔")
    st.dataframe(g.sort_values("買−賣(張)", ascending=False)[
                   ["訊號", "代號", "名稱", "市場", "盤後成交(張)", "盤後占盤中%", "收盤價",
                    "未成交買量(張)", "未成交賣量(張)", "買−賣(張)", "日期"]],
                 hide_index=True, width="stretch", column_config=FMT)
    if missing:
      st.caption(f"查無盤後定價：{'、'.join(missing[:40])}{' …' if len(missing) > 40 else ''}"
                 "（上櫃只有最新一天、或當天沒有交易）")

  with tab_mkt:
    cs = mk[mk["代號"].map(is_common_stock)]
    show = ["市場", "代號", "名稱", "盤後成交(張)", "盤後占盤中%", "收盤價", "未成交買量(張)", "未成交賣量(張)"]
    min_lots = st.number_input("「盤後占比」排行的最低盤後成交（張）", min_value=0, value=100, step=50, key="ah_min")
    r1, r2, r3 = st.columns(3)
    with r1:
      st.markdown("**🔴 收盤價買不到**（未成交買量最多）")
      st.dataframe(cs[cs["買−賣(張)"] > 0].nlargest(20, "未成交買量(張)")[show],
                   hide_index=True, width="stretch", column_config=FMT)
    with r2:
      st.markdown("**🟢 收盤價賣不掉**（未成交賣量最多）")
      st.dataframe(cs[cs["買−賣(張)"] < 0].nlargest(20, "未成交賣量(張)")[show],
                   hide_index=True, width="stretch", column_config=FMT)
    with r3:
      st.markdown(f"**盤後占盤中比例最高**（盤後 ≥ {min_lots} 張）")
      st.dataframe(cs[cs["盤後成交(張)"] >= min_lots].nlargest(20, "盤後占盤中%")[show],
                   hide_index=True, width="stretch", column_config=FMT)
    st.caption("只列普通股（排除 ETF、權證、特別股）。")

  with tab_one:
    a, b = st.columns([2, 3])
    code = a.text_input("股票代號", value="2330", key="ah_code").strip().upper()
    n_days = b.select_slider("上市股看最近幾個交易日", options=[5, 10, 20], value=10, key="ah_days",
                             help="每個交易日要向證交所抓一次全市場資料，第一次比較慢，之後有快取")
    if code:
      in_tpex = code in set(tpex["代號"])
      hist = pd.DataFrame(columns=COLS) if in_tpex else twse_history(code, n_days)
      if hist.empty and in_tpex:
        hist = tpex[tpex["代號"] == code].reset_index(drop=True)
        st.caption("上櫃股票：櫃買中心的公開資料只提供最新一個交易日，所以沒有歷史走勢。")
      if hist.empty:
        st.warning(f"查無 {code} 的盤後定價資料。")
      else:
        lr = hist.iloc[0]
        st.caption(f"最新資料日：**{lr['日期']}**（{lr['市場']}）　{names.get(code, lr['名稱'])}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("盤後成交", f"{lr['盤後成交(張)']:,.0f} 張")
        k2.metric("占盤中", "—" if pd.isna(lr["盤後占盤中%"]) else f"{lr['盤後占盤中%']:.2f}%")
        k3.metric("撮合價", f"{lr['收盤價']:,.2f}", help="盤後定價一律以當日收盤價成交")
        k4.metric("未成交買", f"{lr['未成交買量(張)']:,.0f} 張")
        k5.metric("未成交賣", f"{lr['未成交賣量(張)']:,.0f} 張")
        if len(hist) > 1:
          st.plotly_chart(_history_chart(hist))
        st.dataframe(hist.drop(columns=["代號", "名稱"]), hide_index=True, width="stretch", column_config=FMT)

  st.caption("資料來源：證交所、櫃買中心公開資料。教學與研究用途，不構成投資建議。")
