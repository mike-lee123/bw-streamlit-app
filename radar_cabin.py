"""第 23 艙「📡 法說會籌碼雷達艙」。

資料來自本機 n8n 的「法說會籌碼雷達」與「打臉追蹤」流程，每天由 export_to_github.py 推到本 repo 的
radar-data 分支（radar.json、scorecard.json、broker_news.json、meta.json）。這裡只讀取、顯示、提供下載。

網址參數：?cabin=radar&stock=3231 會直接打開本艙並聚焦在該股（雷達 Email 的連結就是這種格式）。
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

DATA_URL = "https://raw.githubusercontent.com/mike-lee123/bw-streamlit-app/radar-data/{}"


@st.cache_data(ttl=900, show_spinner=False)
def load_json(name: str):
  r = requests.get(DATA_URL.format(name), timeout=20)
  r.raise_for_status()
  return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(stock_id: str, market: str) -> pd.DataFrame:
  suffixes = [".TWO", ".TW"] if market == "上櫃" else [".TW", ".TWO"]
  for suf in suffixes:
    try:
      df = yf.download(stock_id + suf, period="6mo", interval="1d", progress=False, auto_adjust=False)
    except Exception:
      continue
    if df is not None and not df.empty:
      if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
      return df[["Close"]].dropna()
  return pd.DataFrame()


def _fmt(v, unit="", digits=1, sign=True):
  if v is None or (isinstance(v, float) and pd.isna(v)):
    return "—"
  v = float(v)
  s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
  return s + unit


# ---------- 外部內容的安全處理 ----------
# 新聞和 PTT 標題來自外部網站，顯示前要跳脫 Markdown 符號，連結只放行預期的網域
_ALLOWED_HOSTS = ("news.google.com", "www.ptt.cc", "mopsov.twse.com.tw")
_MD_SPECIAL = str.maketrans({ch: "\\" + ch for ch in r"\`*_{}[]()<>#+-.!|~"})


def _md(text) -> str:
  return str(text or "").translate(_MD_SPECIAL)


def _safe_url(url) -> str:
  from urllib.parse import urlparse
  try:
    u = urlparse(str(url or ""))
  except ValueError:
    return ""
  return str(url) if u.scheme == "https" and u.hostname in _ALLOWED_HOSTS else ""


def _link(title, url) -> str:
  safe = _safe_url(url)
  return f"[{_md(title)}]({safe})" if safe else _md(title)


def _no_formula(df: pd.DataFrame) -> pd.DataFrame:
  """下載前處理：以 = + - @ 或 Tab／換行開頭的文字前面加 '，Excel 才不會當成公式執行。數字欄位不受影響。"""
  out = df.copy()
  for col in out.columns:
    if out[col].dtype == object:
      out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v)
  return out


BROKER_TXT = {"up": "以調升為主", "down": "以調降為主", "mixed": "方向不明顯", "flat": "只報價位", "none": "無報導"}


def _flatten(rows: list) -> pd.DataFrame:
  """雷達的巢狀 JSON → 一列一場法說會的表格（給畫面和下載用）。"""
  out = []
  for r in rows:
    c = r.get("chips") or {}
    b = r.get("brokers") or {}
    rt = r.get("retail") or {}
    cs = b.get("consensus") or {}
    ex5 = None if r.get("ret_5d") is None else r["ret_5d"] - (r.get("mkt_ret_5d") or 0)
    ex20 = None if r.get("ret_20d") is None else r["ret_20d"] - (r.get("mkt_ret_20d") or 0)
    out.append({
        "法說會日期": r.get("call_date"), "代號": r.get("stock_id"), "名稱": r.get("name"), "市場": r.get("market"),
        "訊號": r.get("signal"), "等級": r.get("level"),
        "簡報語氣": r.get("tone"), "語氣分數": r.get("tone_score"), "新聞情緒": r.get("news_sentiment"),
        "外資5日(張)": c.get("foreign_5d"), "投信5日(張)": c.get("trust_5d"), "法人合計5日(張)": c.get("inst_5d"),
        "法人佔成交量%": c.get("inst_5d_pct"), "融資5日(張)": c.get("margin_5d"), "收盤價": c.get("close"),
        "股價5日%": c.get("price_5d_pct"),
        "券商動向": BROKER_TXT.get(b.get("trend"), ""), "券商調升": b.get("ups"), "券商調降": b.get("downs"),
        "FactSet共識目標價": cs.get("target"), "共識上漲空間%": b.get("upside_pct"),
        "PTT近7天篇數": rt.get("posts_7d"), "PTT熱度倍數": rt.get("heat_ratio"), "PTT過熱": rt.get("hot"),
        "PTT看多": rt.get("bull"), "PTT看空": rt.get("bear"),
        "5日報酬%": r.get("ret_5d"), "5日大盤%": r.get("mkt_ret_5d"), "5日超額%": ex5,
        "20日報酬%": r.get("ret_20d"), "20日大盤%": r.get("mkt_ret_20d"), "20日超額%": ex20,
        "一句話重點": r.get("one_line"), "展望重點": "；".join(r.get("guidance") or []),
        "風險": "；".join(r.get("risks") or []), "附註": "；".join(r.get("notes") or []),
        "簡報PDF": r.get("pdf_url"), "分析時間": r.get("analyzed_at"),
    })
  return pd.DataFrame(out)


def _excel(sheets: dict) -> bytes:
  buf = io.BytesIO()
  with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    for name, df in sheets.items():
      _no_formula(df).to_excel(xw, sheet_name=name[:31], index=False)
  return buf.getvalue()


def _stock_detail(rows: list, stock_id: str, news: list):
  mine = [r for r in rows if r.get("stock_id") == stock_id]
  if not mine:
    st.info(f"雷達裡還沒有 {stock_id} 的法說會分析。")
    return
  first = mine[0]
  st.markdown(f"#### {first.get('name')}（{stock_id}）")

  prices = load_prices(stock_id, first.get("market") or "")
  if not prices.empty:
    fig = go.Figure(go.Scatter(x=prices.index, y=prices["Close"], mode="lines", name="收盤價", line=dict(color="#1c7ed6")))
    for r in mine:
      fig.add_vline(x=r["call_date"], line_dash="dot", line_color="#e67700")
      fig.add_annotation(x=r["call_date"], y=1, yref="paper", text="法說會", showarrow=False, font=dict(size=11, color="#e67700"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig)

  for r in mine:
    c = r.get("chips") or {}
    b = r.get("brokers") or {}
    rt = r.get("retail") or {}
    with st.container(border=True):
      st.markdown(f"**{r.get('signal')}**　法說會 {r.get('call_date')}")
      if r.get("one_line"):
        st.write(r["one_line"])
      # 用小字排版，不用 st.metric：metric 字太大，窄螢幕會把「非常樂觀」截成「非常…」
      heat = rt.get("heat_ratio")
      cells = [
          ("簡報語氣", f"{r.get('tone') or '—'}（{_fmt(r.get('tone_score'), digits=0)}）", ""),
          ("法人 5 日", _fmt(c.get("inst_5d"), " 張", 0), f"佔成交量 {_fmt(c.get('inst_5d_pct'), '%')}"),
          ("券商動向", BROKER_TXT.get(b.get("trend"), "—"), f"調升 {b.get('ups', 0)}／調降 {b.get('downs', 0)}"),
          ("PTT 散戶", f"近 7 天 {rt.get('posts_7d', 0)} 篇",
           "過熱" if rt.get("hot") else (f"熱度 {heat} 倍" if heat is not None else "")),
      ]
      from html import escape
      for col, (label, value, sub) in zip(st.columns(4), cells):
        col.markdown(f"<div style='font-size:12px;opacity:.65'>{escape(str(label))}</div>"
                     f"<div style='font-size:16px;font-weight:600'>{escape(str(value))}</div>"
                     f"<div style='font-size:12px;opacity:.65'>{escape(str(sub))}</div>", unsafe_allow_html=True)
      if r.get("ret_5d") is not None:
        ex5 = r["ret_5d"] - (r.get("mkt_ret_5d") or 0)
        ex20 = None if r.get("ret_20d") is None else r["ret_20d"] - (r.get("mkt_ret_20d") or 0)
        st.caption(f"結算：5 日 {_fmt(r['ret_5d'], '%')}（大盤 {_fmt(r.get('mkt_ret_5d'), '%')}，超額 {_fmt(ex5, '%')}）"
                   + (f"；20 日 {_fmt(r['ret_20d'], '%')}（超額 {_fmt(ex20, '%')}）" if ex20 is not None else "；20 日尚未結算"))
      cols = st.columns(2)
      with cols[0]:
        if r.get("guidance"):
          st.markdown("**展望重點**\n" + "\n".join(f"- {g}" for g in r["guidance"]))
        if r.get("risks"):
          st.markdown("**風險**\n" + "\n".join(f"- {g}" for g in r["risks"]))
      with cols[1]:
        items = [i for i in (b.get("items") or []) if i.get("direction") != "flat"][:4]
        if items:
          arrow = {"up": "▲", "down": "▼"}
          st.markdown("**券商目標價新聞**\n" + "\n".join(
              f"- {arrow.get(i['direction'], '・')} {_md((i.get('date') or '')[5:])} {_link(i['title'][:36], i.get('link'))}" for i in items))
        if rt.get("titles"):
          st.markdown("**PTT 近期討論**\n" + "\n".join(
              f"- {_md(t['date'])} {_link(t['title'][:36], t.get('url'))}（{'爆' if t['nrec'] >= 100 else int(t['nrec'])} 推）" for t in rt["titles"][:3]))
      if r.get("notes"):
        st.caption(_md("；".join(r["notes"])))
      if _safe_url(r.get("pdf_url")):
        st.markdown(f"[📄 法說會簡報 PDF]({_safe_url(r['pdf_url'])})")


def render_radar_cabin(focus_stock: str | None = None):
  st.write("### 📡 法說會籌碼雷達艙")
  st.caption("每個交易日 17:30 由本機 n8n 分析自選股法說會：AI 讀簡報判斷語氣，程式比對法人籌碼、券商目標價新聞與 PTT 散戶情緒，"
             "挑出「說的」和「做的」不一致的股票。研究參考，不構成投資建議。")
  try:
    rows = load_json("radar.json")
    score = load_json("scorecard.json")
    news = load_json("broker_news.json")
    meta = load_json("meta.json")
  except Exception as e:
    st.error(f"讀不到雷達資料（{e}）。資料由本機每天推送，電腦或 Docker 沒開時不會更新。")
    return
  st.caption(f"資料更新：{meta.get('updated_at', '—')}　｜　雷達累計 {len(rows)} 場")

  df = _flatten(rows)
  if df.empty:
    st.info("雷達還沒有資料。")
    return

  # ---------- 篩選 ----------
  f1, f2, f3 = st.columns([1, 1, 2])
  days = f1.selectbox("期間", ["近 7 天", "近 30 天", "近 90 天", "全部"], index=1, key="radar_days")
  only_alert = f2.checkbox("只看值得注意", value=False, key="radar_alert")
  stock_opts = ["（全部）"] + sorted({f"{r['stock_id']} {r['name']}" for r in rows})
  focus_label = next((o for o in stock_opts if focus_stock and o.split(" ")[0] == focus_stock), "（全部）")
  stock_sel = f3.selectbox("個股", stock_opts, index=stock_opts.index(focus_label), key="radar_stock")

  view = df.copy()
  if stock_sel == "（全部）":
    n = {"近 7 天": 7, "近 30 天": 30, "近 90 天": 90}.get(days)
    if n:
      cutoff = (pd.Timestamp.now(tz="Asia/Taipei") - pd.Timedelta(days=n)).strftime("%Y-%m-%d")
      view = view[view["法說會日期"] >= cutoff]
  else:
    view = view[view["代號"] == stock_sel.split(" ")[0]]
  if only_alert:
    view = view[view["等級"] >= 2]

  k = st.columns(4)
  k[0].metric("法說會場數", len(view))
  k[1].metric("值得注意", int((view["等級"] >= 2).sum()))
  k[2].metric("🔴 背離警報", int(view["訊號"].str.startswith("🔴").sum()))
  k[3].metric("已結算（5 日）", int(view["5日報酬%"].notna().sum()))

  show_cols = ["法說會日期", "代號", "名稱", "訊號", "簡報語氣", "新聞情緒", "法人合計5日(張)", "法人佔成交量%",
               "券商動向", "PTT熱度倍數", "5日超額%", "20日超額%"]
  st.dataframe(view[show_cols], width="stretch", hide_index=True)

  # ---------- 下載 ----------
  sig = pd.DataFrame(score.get("signals") or [])
  brk = pd.DataFrame(score.get("brokers") or [])
  nws = pd.DataFrame(news or [])
  d1, d2 = st.columns(2)
  d1.download_button("⬇️ 下載 CSV（目前篩選）", _no_formula(view).to_csv(index=False).encode("utf-8-sig"),
                     file_name=f"法說會雷達_{meta.get('updated_at', '')[:10]}.csv", mime="text/csv", key="radar_csv")
  d2.download_button("⬇️ 下載 Excel（雷達＋戰績＋券商新聞）",
                     _excel({"雷達訊號": view, "訊號戰績": sig, "券商戰績": brk, "券商目標價新聞": nws}),
                     file_name=f"法說會雷達_{meta.get('updated_at', '')[:10]}.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="radar_xlsx")

  # ---------- 個股細節 ----------
  st.markdown("---")
  detail_opts = sorted({f"{a} {b}" for a, b in zip(view["代號"], view["名稱"])})
  if detail_opts:
    default = stock_sel if stock_sel in detail_opts else detail_opts[0]
    pick = st.selectbox("🔍 個股細節", detail_opts, index=detail_opts.index(default), key="radar_detail")
    _stock_detail(rows, pick.split(" ")[0], news)

  # ---------- 打臉戰績 ----------
  st.markdown("---")
  st.markdown("#### 🎯 打臉戰績")
  st.caption("命中看「超額報酬」（個股減加權指數）：看空訊號跑輸大盤、看多訊號跑贏大盤才算命中。樣本少於 10 筆時參考價值有限。")
  if not sig.empty:
    sig_v = sig.rename(columns={"signal_type": "訊號", "bias": "方向", "n_5d": "5日樣本", "avg_excess_5d": "5日平均超額%",
                                "hit_rate_5d": "5日命中率%", "n_20d": "20日樣本", "avg_excess_20d": "20日平均超額%",
                                "hit_rate_20d": "20日命中率%"})
    sig_v["方向"] = sig_v["方向"].map({"bear": "看空", "bull": "看多", "none": "—"})
    st.dataframe(sig_v, width="stretch", hide_index=True)
  if not brk.empty:
    brk_v = brk.rename(columns={"broker": "券商", "direction": "方向", "n_20d": "20日樣本", "avg_excess_20d": "20日平均超額%",
                                "hit_rate_20d": "20日命中率%", "n_total": "總則數"})
    brk_v["方向"] = brk_v["方向"].map({"up": "▲ 調升", "down": "▼ 調降"})
    with st.expander("各券商目標價調整的戰績"):
      st.dataframe(brk_v, width="stretch", hide_index=True)
