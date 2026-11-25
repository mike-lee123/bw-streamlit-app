"""第 25 艙「💱 台幣匯率×台股艙」。

USD/TWD 技術分析（K 線、MA20/MA60、布林、MACD 交叉、RSI）＋ 股匯連動（台股 vs 台幣、滾動相關、雙 MACD、報酬散佈圖）。
資料來源 Yahoo Finance（TWD=X 與台股代號），每小時快取。即期外匯是店頭市場、Yahoo 無成交量，以日振幅替代。
由獨立的「台幣匯率 × 台股 技術分析儀表板」（fx_core.py + app.py）併入；設定放在艙內，不佔用左側控制台。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

UP, DN = "#e03131", "#2f9e44"   # 台灣習慣：紅漲綠跌
FX_TICKER = "TWD=X"

FX_STOCKS = {
  "加權指數 ^TWII": "^TWII",
  "台積電 2330": "2330.TW",
  "鴻海 2317": "2317.TW",
  "聯發科 2454": "2454.TW",
  "元大台灣50 0050": "0050.TW",
}
FX_PERIODS = {"6 個月": "6mo", "1 年": "1y", "2 年": "2y", "5 年": "5y"}


# ---------- 資料 ----------
def fx_fetch(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
  df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
  if df is None or df.empty:
    raise ValueError(f"抓不到 {ticker} 資料")
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
  return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ---------- 指標 ----------
def fx_macd(close: pd.Series, fast=12, slow=26, signal=9):
  dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
  dea = dif.ewm(span=signal, adjust=False).mean()
  return dif, dea, dif - dea


def fx_rsi(close: pd.Series, n=14) -> pd.Series:
  d = close.diff()
  gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
  loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
  return 100 - 100 / (1 + gain / loss)


def fx_add_indicators(df: pd.DataFrame) -> pd.DataFrame:
  c = df["Close"]
  df["MA20"] = c.rolling(20).mean()
  df["MA60"] = c.rolling(60).mean()
  std = c.rolling(20).std()
  df["BB_up"] = df["MA20"] + 2 * std
  df["BB_dn"] = df["MA20"] - 2 * std
  df["DIF"], df["DEA"], df["HIST"] = fx_macd(c)
  df["RSI"] = fx_rsi(c)
  return df


@st.cache_data(ttl=3600, show_spinner="下載匯率資料…")
def fx_build_fx(period: str, interval: str) -> pd.DataFrame:
  df = fx_add_indicators(fx_fetch(FX_TICKER, period, interval))
  df["Range"] = df.High - df.Low   # 即期外匯無成交量，以日振幅替代
  return df


@st.cache_data(ttl=3600, show_spinner="下載台股資料…")
def fx_build_link(stock: str, period: str, corr_window: int) -> pd.DataFrame:
  stk = fx_fetch(stock, period)
  fx = fx_fetch(FX_TICKER, period)
  # 只保留台股有開盤的日子（匯率在台股休市日也有報價）
  df = stk.join(fx[["Close"]].rename(columns={"Close": "FX"}), how="inner")
  df = fx_add_indicators(df)
  df["F_DIF"], df["F_DEA"], df["F_HIST"] = fx_macd(df.FX)
  r_stk = df.Close.pct_change()
  r_twd = -df.FX.pct_change()          # 正＝台幣升值
  df["R_STK"], df["R_TWD"] = r_stk, r_twd
  df["CORR"] = r_stk.rolling(corr_window).corr(r_twd)
  return df


def fx_cross_points(df: pd.DataFrame, prefix=""):
  x = (df[prefix + "DIF"] > df[prefix + "DEA"]).astype(int).diff()
  return df.index[x == 1], df.index[x == -1]


def fx_pct_change(s: pd.Series, n: int) -> float:
  return (s.iloc[-1] / s.iloc[-n - 1] - 1) * 100 if len(s) > n else float("nan")


# ---------- 圖表 ----------
def _fx_colors(values):
  return [UP if v >= 0 else DN for v in values]


def _fx_layout(fig, height):
  fig.update_layout(height=height, template="plotly_dark", hovermode="x unified",
                    xaxis_rangeslider_visible=False, margin=dict(t=40, b=10, l=10, r=10),
                    legend=dict(orientation="h", y=1.02, x=0))
  fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
  return fig


def _fx_macd_panel(fig, df, row, prefix=""):
  fig.add_trace(go.Bar(x=df.index, y=df[prefix + "HIST"], marker_color=_fx_colors(df[prefix + "HIST"].fillna(0)),
                       name="MACD 柱", showlegend=False), row, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df[prefix + "DIF"], name="DIF", showlegend=False,
                           line=dict(color="#f08c00", width=1.2)), row, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df[prefix + "DEA"], name="DEA", showlegend=False,
                           line=dict(color="#1c7ed6", width=1.2)), row, 1)


def fx_figure(df: pd.DataFrame, show_bb=True, show_cross=True) -> go.Figure:
  fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                      row_heights=[0.5, 0.13, 0.22, 0.15],
                      subplot_titles=("USD/TWD（上漲＝台幣貶值）", "日振幅 高-低（即期外匯無成交量，替代指標）",
                                      "MACD (12, 26, 9)", "RSI (14)"))
  fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                               name="USD/TWD", increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
  lines = [("MA20", "#f08c00", None), ("MA60", "#1c7ed6", None)]
  if show_bb:
    lines += [("BB_up", "#adb5bd", "dot"), ("BB_dn", "#adb5bd", "dot")]
  for col, color, dash in lines:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                             line=dict(color=color, width=1.2, dash=dash)), 1, 1)
  if show_cross:
    g, d = fx_cross_points(df)
    fig.add_trace(go.Scatter(x=g, y=df.loc[g, "Low"] * 0.998, mode="markers", name="黃金交叉",
                             marker=dict(symbol="triangle-up", size=10, color=UP)), 1, 1)
    fig.add_trace(go.Scatter(x=d, y=df.loc[d, "High"] * 1.002, mode="markers", name="死亡交叉",
                             marker=dict(symbol="triangle-down", size=10, color=DN)), 1, 1)
  fig.add_trace(go.Bar(x=df.index, y=df.Range, showlegend=False, name="振幅",
                       marker_color=[UP if c >= o else DN for o, c in zip(df.Open, df.Close)]), 2, 1)
  _fx_macd_panel(fig, df, 3)
  fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", showlegend=False,
                           line=dict(color="#7048e8", width=1.3)), 4, 1)
  for lvl in (70, 30):
    fig.add_hline(y=lvl, line_dash="dash", line_color="#adb5bd", row=4, col=1)
  return _fx_layout(fig, 900)


def fx_link_figure(df: pd.DataFrame, name: str, corr_window: int) -> go.Figure:
  fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.28, 0.10, 0.20, 0.12, 0.15, 0.15],
    specs=[[{}], [{}], [{"secondary_y": True}], [{}], [{}], [{}]],
    subplot_titles=(f"{name} K 線", f"{name} 成交量",
                    f"{name}（左軸）vs USD/TWD（右軸反轉：往上＝台幣升值）",
                    f"{corr_window} 日滾動相關（股市報酬 vs 台幣升值）",
                    f"{name} MACD", "USD/TWD MACD（柱紅＝台幣轉弱）"))
  fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                               name=name, increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name="MA20", line=dict(color="#f08c00", width=1.2)), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.MA60, name="MA60", line=dict(color="#1c7ed6", width=1.2)), 1, 1)
  fig.add_trace(go.Bar(x=df.index, y=df.Volume, showlegend=False, name="成交量",
                       marker_color=[UP if c >= o else DN for o, c in zip(df.Open, df.Close)]), 2, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.Close, name=f"{name} 收盤",
                           line=dict(color="#ff6b6b", width=1.6)), 3, 1, secondary_y=False)
  fig.add_trace(go.Scatter(x=df.index, y=df.FX, name="USD/TWD",
                           line=dict(color="#4dabf7", width=1.6)), 3, 1, secondary_y=True)
  fig.update_yaxes(autorange="reversed", secondary_y=True, row=3, col=1)
  fig.add_trace(go.Bar(x=df.index, y=df.CORR, marker_color=_fx_colors(df.CORR.fillna(0)),
                       showlegend=False, name="相關"), 4, 1)
  fig.update_yaxes(range=[-1, 1], row=4, col=1)
  _fx_macd_panel(fig, df, 5)
  _fx_macd_panel(fig, df, 6, prefix="F_")
  return _fx_layout(fig, 1250)


def fx_scatter_figure(df: pd.DataFrame, name: str) -> go.Figure:
  d = df[["R_TWD", "R_STK"]].dropna() * 100
  fig = go.Figure(go.Scatter(x=d.R_TWD, y=d.R_STK, mode="markers",
                             marker=dict(size=6, color="#4dabf7", opacity=0.55),
                             text=d.index.strftime("%Y-%m-%d"),
                             hovertemplate="%{text}<br>台幣 %{x:+.2f}%<br>股市 %{y:+.2f}%<extra></extra>"))
  fig.add_hline(y=0, line_color="#868e96")
  fig.add_vline(x=0, line_color="#868e96")
  fig.update_layout(template="plotly_dark", height=480, margin=dict(t=30, b=10, l=10, r=10),
                    xaxis_title="台幣日報酬 %（正＝升值）", yaxis_title=f"{name} 日報酬 %")
  return fig


# ---------- 訊號解讀 ----------
def _fx_trend(row) -> str:
  p, m20, m60 = row.Close, row.MA20, row.MA60
  if p > m20 > m60:
    return "多頭排列"
  if p < m20 < m60:
    return "空頭排列"
  return "盤整"


def _fx_momentum(hist_now, hist_prev) -> str:
  if hist_now >= 0:
    return "多方動能增強" if hist_now > hist_prev else "多方動能減弱"
  return "空方動能增強" if hist_now < hist_prev else "空方動能減弱"


# ---------- 介面 ----------
def render_fx_cabin():
  """第 25 艙：台幣匯率 × 台股 技術分析"""
  st.header("💱 台幣匯率 × 台股 技術分析")

  with st.expander("⚙️ 本艙設定", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    period_label = c1.selectbox("期間", list(FX_PERIODS), index=2, key="fx_period")
    interval = c2.radio("匯率 K 線週期", ["1d", "1wk"], horizontal=True, key="fx_interval",
                        format_func={"1d": "日線", "1wk": "週線"}.get)
    stock_label = c3.selectbox("台股標的", list(FX_STOCKS), key="fx_stock")
    custom = c4.text_input("或自訂代號（如 2603.TW）", key="fx_custom").strip()
    c5, c6, c7, c8 = st.columns(4)
    corr_window = c5.slider("滾動相關天數", 20, 120, 60, step=10, key="fx_corr")
    show_bb = c6.checkbox("顯示布林通道", True, key="fx_bb")
    show_cross = c7.checkbox("標示 MACD 交叉", True, key="fx_cross")
    if c8.button("🔄 重新抓取資料", key="fx_refresh", width="stretch"):
      fx_build_fx.clear()
      fx_build_link.clear()
    st.caption("資料來源：Yahoo Finance，每小時快取一次。僅供技術分析參考，不構成投資建議。")

  period = FX_PERIODS[period_label]
  stock = custom.upper() or FX_STOCKS[stock_label]
  name = "加權指數" if stock == "^TWII" else (custom.upper() or stock_label)

  try:
    fx = fx_build_fx(period, interval)
    link = fx_build_link(stock, period, corr_window)
  except Exception as e:  # 網路、Yahoo 限流或代號錯誤
    st.error(f"讀取資料失敗：{e}（Yahoo 偶爾限流，稍等幾分鐘再按「重新抓取資料」）")
    return
  if len(fx) < 3 or len(link) < 3:
    st.warning("資料筆數太少，請換長一點的期間。")
    return

  st.caption(f"最新資料日：匯率 {fx.index[-1]:%Y-%m-%d}｜{name} {link.index[-1]:%Y-%m-%d}　"
             "（紅漲綠跌；USD/TWD 上漲＝台幣貶值）")

  f_last, s_last = fx.iloc[-1], link.iloc[-1]
  k1, k2, k3, k4, k5 = st.columns(5)
  k1.metric("USD/TWD", f"{f_last.Close:.3f}",
            f"{f_last.Close - fx.Close.iloc[-2]:+.3f}", delta_color="inverse", border=True)
  k2.metric(name, f"{s_last.Close:,.2f}",
            f"{fx_pct_change(link.Close, 1):+.2f}%", delta_color="inverse", border=True)
  k3.metric("20日台幣", f"{-fx_pct_change(link.FX, 20):+.2f}%", help="正＝台幣升值", border=True)
  k4.metric("20日股價", f"{fx_pct_change(link.Close, 20):+.2f}%", border=True)
  k5.metric("股匯相關", f"{s_last.CORR:+.2f}",
            help="股市日報酬 vs 台幣升值日報酬；正＝台幣升值時股市同步上漲", border=True)

  with st.container(border=True):
    st.subheader("📋 訊號摘要")
    fx_trend = _fx_trend(f_last)
    fx_mom = _fx_momentum(f_last.HIST, fx.HIST.iloc[-2])
    s_trend = _fx_trend(s_last)
    s_mom = _fx_momentum(s_last.HIST, link.HIST.iloc[-2])
    twd_word = {"多頭排列": "台幣偏貶", "空頭排列": "台幣偏升", "盤整": "台幣區間整理"}[fx_trend]
    rsi_note = "超買" if f_last.RSI > 70 else "超賣" if f_last.RSI < 30 else "中性"
    a, b = st.columns(2)
    a.markdown(f"**USD/TWD**：{fx_trend}（{twd_word}），MACD {fx_mom}，"
               f"RSI {f_last.RSI:.1f}（{rsi_note}）")
    b.markdown(f"**{name}**：{s_trend}，MACD {s_mom}，RSI {s_last.RSI:.1f}")
    corr = s_last.CORR
    if pd.isna(corr):
      st.info(f"資料不足 {corr_window} 天，尚無法計算滾動相關。")
    elif corr > 0.3:
      st.info("股匯連動偏強：台幣升值時股市傾向上漲，外資資金流向可能是主要推力。")
    elif corr < -0.1:
      st.info("股匯呈反向關係：近期股市漲跌與台幣走勢背離，可能有內資或其他因素主導。")
    else:
      st.info("股匯連動偏弱：近期匯率對股市的解釋力有限。")

  t1, t2, t3, t4 = st.tabs(["💵 匯率技術分析", "📈 股匯連動", "🔍 報酬散佈圖", "🗂️ 資料表"])

  with t1:
    st.plotly_chart(fx_figure(fx, show_bb, show_cross))
    st.caption("即期外匯為店頭市場，Yahoo 無成交量，第二面板以日振幅（高-低）替代。")

  with t2:
    st.plotly_chart(fx_link_figure(link, name, corr_window))

  with t3:
    st.plotly_chart(fx_scatter_figure(link, name))
    full = link[["R_STK", "R_TWD"]].corr().iloc[0, 1]
    st.caption(f"全期相關係數 {full:+.2f}。右上象限＝台幣升值且股市上漲；左下＝台幣貶值且股市下跌。")

  with t4:
    cols = ["Open", "High", "Low", "Close", "Volume", "MA20", "MA60",
            "DIF", "DEA", "HIST", "RSI", "FX", "F_HIST", "CORR"]
    table = link[cols].sort_index(ascending=False)
    st.dataframe(table, width="stretch", height=500)
    st.download_button("⬇️ 下載 CSV", table.to_csv().encode("utf-8-sig"),
                       file_name=f"fx_{stock}_{period}.csv", mime="text/csv", key="fx_dl")
