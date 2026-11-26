"""第 25 艙「💱 台幣匯率×台股艙」。

USD/TWD 技術分析（K 線、MA20/MA60、布林、MACD 交叉、RSI）＋ 股匯連動（台股 vs 台幣、滾動相關、雙 MACD、報酬散佈圖）
＋ 主力痕跡・威科夫（開高低收量分工：收盤位置、影線、量比、布林壓縮、彈簧／上衝回落、背離、階段底色、多方轉折檢查表）。
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
  df = df[["Open", "High", "Low", "Close", "Volume"]]
  if ticker == "^TWII" and interval == "1d":
    df = _patch_twii(df)
  return df.dropna()


def _twse_taiex_month(month_first: pd.Timestamp) -> pd.DataFrame:
  """證交所官方加權指數日線（一次一個月）。日期是民國年，例如 115/11/25。"""
  import requests
  r = requests.get("https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
                   params={"date": month_first.strftime("%Y%m01"), "response": "json"},
                   headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
  j = r.json()
  if j.get("stat") != "OK" or not j.get("data"):
    return pd.DataFrame()
  rows = []
  for d, o, h, l, c in j["data"]:
    y, m, dd = d.split("/")
    rows.append({"Date": pd.Timestamp(int(y) + 1911, int(m), int(dd)),
                 "Open": float(o.replace(",", "")), "High": float(h.replace(",", "")),
                 "Low": float(l.replace(",", "")), "Close": float(c.replace(",", "")), "Volume": 0.0})
  return pd.DataFrame(rows).set_index("Date")


def _patch_twii(df: pd.DataFrame) -> pd.DataFrame:
  """Yahoo 的加權指數偶爾會缺最近幾天（整列沒有，或收盤是空值）。
  用證交所官方資料補上最近兩個月缺的日子；證交所連不上就維持 Yahoo 原樣。
  （補上的日子沒有成交量，以 0 填入；本專案的指數指標不使用成交量。）"""
  try:
    today = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    this_month = today.replace(day=1)
    twse = pd.concat([_twse_taiex_month(this_month - pd.DateOffset(months=1)), _twse_taiex_month(this_month)])
  except Exception:  # noqa: BLE001
    return df
  if twse.empty:
    return df
  missing = [d for d in twse.index if d not in df.index or pd.isna(df.at[d, "Close"])]
  if not missing:
    return df
  df = df.drop(index=[d for d in missing if d in df.index])
  return pd.concat([df, twse.loc[missing]]).sort_index()


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


# ---------- 主力痕跡・威科夫（開高低收量分工） ----------
FX_WK_PHASES = {   # 階段: (底色, 圖例色)
  "吸籌": ("rgba(77,171,247,0.13)", "rgba(77,171,247,0.8)"),
  "拉抬": ("rgba(255,107,107,0.12)", "rgba(255,107,107,0.8)"),
  "出貨": ("rgba(255,169,77,0.14)", "rgba(255,169,77,0.8)"),
  "收割": ("rgba(81,207,102,0.12)", "rgba(81,207,102,0.8)"),
}
FX_WK_EVENTS = {
  "Spring": "彈簧：假跌破洗盤",
  "Upthrust": "上衝回落：假突破誘多",
  "Climax": "爆量長影線",
  "BullDiv": "MACD 底背離",
  "BearDiv": "MACD 頂背離",
  "VolDiv": "創高量縮",
  "Squeeze": "布林壓縮",
}


def _fx_first_in(mask: pd.Series, days: int = 10) -> pd.Series:
  """同一訊號 days 天內只保留第一次。"""
  seen = mask.shift().rolling(days, min_periods=1).max().fillna(0).astype(bool)
  return mask & ~seen


def fx_add_wyckoff(df: pd.DataFrame, lookback: int = 20, squeeze_window: int = 120) -> pd.DataFrame:
  """在 fx_build_link 的結果上加入 OHLCV 分工指標與簡化威科夫階段。"""
  df = df.copy()
  rng = (df.High - df.Low).replace(0, float("nan"))
  body_top = df[["Open", "Close"]].max(axis=1)
  body_bot = df[["Open", "Close"]].min(axis=1)
  df["CLV"] = (df.Close - df.Low) / rng                 # 收盤＝主力表態
  df["UpShadow"] = (df.High - body_top) / rng           # 上方流動性被吃
  df["LowShadow"] = (body_bot - df.Low) / rng           # 下方停損被掃
  vol = df.Volume.where(df.Volume > 0)                  # 盤中 Yahoo 量為 0，視為缺值
  df["VolRatio"] = vol / vol.rolling(20, min_periods=10).mean()
  df["BBW"] = (df.BB_up - df.BB_dn) / df.MA20 * 100     # 非線性：波動壓縮
  df["Squeeze"] = df.BBW <= df.BBW.rolling(squeeze_window, min_periods=40).quantile(0.10)

  prev_hi = df.High.rolling(lookback).max().shift()
  prev_lo = df.Low.rolling(lookback).min().shift()
  df["Spring"] = (df.Low < prev_lo) & (df.Close > prev_lo) & (df.CLV > 0.5)
  df["Upthrust"] = (df.High > prev_hi) & (df.Close < prev_hi) & (df.CLV < 0.5)
  df["Climax"] = (df.VolRatio >= 1.8) & ((df.UpShadow >= 0.4) | (df.LowShadow >= 0.4))

  new_hi = df.Close >= df.Close.rolling(lookback).max()
  new_lo = df.Close <= df.Close.rolling(lookback).min()
  dif_peak = df.DIF.rolling(40).max().shift(5)
  dif_trough = df.DIF.rolling(40).min().shift(5)
  df["BearDiv"] = _fx_first_in(new_hi & (dif_peak > 0) & (df.DIF < dif_peak * 0.8) & (df.HIST < 0))
  df["BullDiv"] = _fx_first_in(new_lo & (dif_trough < 0) & (df.DIF > dif_trough * 0.8) & (df.HIST > 0))
  df["VolDiv"] = new_hi & (df.VolRatio < 0.9)

  # 階段：均線排列判方向（線性）；橫盤時依 MA60 斜率判斷是跌後吸籌或漲後出貨
  ma60_slope = df.MA60.diff(20)
  ma20_slope = df.MA20.diff(5)
  up = (df.Close > df.MA20) & (df.MA20 > df.MA60) & (ma20_slope > 0)
  dn = (df.Close < df.MA20) & (df.MA20 < df.MA60) & (ma20_slope < 0)
  phase = pd.Series(pd.NA, index=df.index, dtype="object")
  phase[up] = "拉抬"
  phase[dn] = "收割"
  rest = ~(up | dn) & ma60_slope.notna()
  phase[rest & (ma60_slope >= 0)] = "出貨"
  phase[rest & (ma60_slope < 0)] = "吸籌"
  run = phase.groupby((phase != phase.shift()).cumsum()).transform("size")
  df["Phase"] = phase.where(run >= 3).ffill().fillna("")   # 連續 3 天才確認
  return df


def fx_wk_checklist(df: pd.DataFrame, window: int = 10) -> list:
  """多方轉折檢查表：[(條件, 成立?, 說明)]。"""
  last, prev = df.iloc[-1], df.iloc[-2]
  recent = df.tail(window)
  sig = recent.Spring | (recent.Climax & (recent.LowShadow >= 0.4) & (recent.CLV > 0.5))
  sig_days = recent.index[sig]
  if len(sig_days):
    base = df.loc[sig_days[-1]]
    after = df.loc[sig_days[-1]:].iloc[1:]
    vr = after.VolRatio.dropna()
    retest_ok = len(after) >= 2 and after.Low.min() >= base.Low and len(vr) > 0 and vr.mean() < 1.0
    vr_text = f"{vr.mean():.2f}" if len(vr) else "—"
    retest_note = (f"訊號日後 {len(after)} 天，最低 {after.Low.min():,.2f} vs 訊號低 {base.Low:,.2f}，"
                   f"均量比 {vr_text}（需 ≥2 天且 <1）") if len(after) else "訊號就在最後一天，等待回測"
  else:
    retest_ok, retest_note = False, "尚無訊號日可回測"
  return [
    ("台幣動能轉強（外資停止撤出）", bool(last.F_HIST < 0),
     f"USD/TWD MACD 柱 {last.F_HIST:+.4f}，需 < 0（美元轉弱＝台幣轉強）"),
    (f"近 {window} 日出現彈簧或爆量長下影（收上半部）", bool(len(sig_days)),
     "、".join(d.strftime("%m/%d") for d in sig_days) or "尚未出現"),
    ("回測量縮且不破訊號低點", bool(retest_ok), retest_note),
    ("MACD 空方動能收斂", bool(last.HIST > prev.HIST),
     f"MACD 柱 {last.HIST:+.2f}（前一日 {prev.HIST:+.2f}）"),
    ("收盤守在 MA20 之上", bool(last.Close > last.MA20),
     f"收盤 {last.Close:,.2f} vs MA20 {last.MA20:,.2f}"),
  ]


def fx_wk_events(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
  rows = []
  for dt, r in df.tail(n).iterrows():
    for col, text in FX_WK_EVENTS.items():
      if bool(r[col]):
        rows.append({"日期": dt.strftime("%Y-%m-%d"), "事件": text, "階段": r.Phase,
                     "收盤": round(r.Close, 2), "收盤位置": round(r.CLV, 2),
                     "上影": round(r.UpShadow, 2), "下影": round(r.LowShadow, 2),
                     "量比": round(r.VolRatio, 2)})
  return pd.DataFrame(rows[::-1])


def fx_wyckoff_figure(df: pd.DataFrame, name: str) -> go.Figure:
  fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                      row_heights=[0.40, 0.14, 0.16, 0.15, 0.15],
                      subplot_titles=(f"{name}：威科夫階段 ＋ 主力痕跡",
                                      "收盤位置 CLV（收盤＝主力表態；>0.5 買方掌控）",
                                      "影線比例（上＝上方流動性被吃；下＝停損被掃）",
                                      "量比（成交量 ÷ 20 日均量；紫色 ≥ 1.8 爆量）",
                                      "布林帶寬 %（橘點＝壓縮，非線性變盤前兆）"))
  # K 線必須是第一條 trace，x 軸才會被判定為日期軸
  fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                               name=name, showlegend=False,
                               increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
  for p, (_, solid) in FX_WK_PHASES.items():   # 階段圖例
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[float("nan")], mode="markers", name=p,
                             marker=dict(size=12, symbol="square", color=solid)), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.MA20, name="MA20", line=dict(color="#f08c00", width=1.1)), 1, 1)
  fig.add_trace(go.Scatter(x=df.index, y=df.MA60, name="MA60", line=dict(color="#4dabf7", width=1.1)), 1, 1)
  marks = [("Spring", "Low", 0.985, "triangle-up", UP, "彈簧（假跌破）"),
           ("Upthrust", "High", 1.015, "triangle-down", DN, "上衝回落（假突破）"),
           ("BullDiv", "Low", 0.97, "star", "#ff8787", "MACD 底背離"),
           ("BearDiv", "High", 1.03, "star", "#69db7c", "MACD 頂背離"),
           ("Climax", "High", 1.045, "circle-open", "#b197fc", "爆量長影線")]
  for col, ref, k, sym, color, label in marks:
    d = df[df[col]]
    fig.add_trace(go.Scatter(x=d.index, y=d[ref] * k, mode="markers", name=label,
                             marker=dict(symbol=sym, size=10, color=color,
                                         line=dict(width=1.5, color=color))), 1, 1)
  # 階段底色（須在子圖已有 trace 後加入，否則 plotly 會略過）
  seg = (df.Phase != df.Phase.shift()).cumsum()
  for _, g in df.groupby(seg):
    p = g.Phase.iloc[0]
    if p in FX_WK_PHASES:
      fig.add_vrect(x0=g.index[0], x1=g.index[-1], fillcolor=FX_WK_PHASES[p][0],
                    line_width=0, layer="below", row=1, col=1)

  fig.add_trace(go.Bar(x=df.index, y=df.CLV, name="CLV", showlegend=False,
                       marker_color=[UP if v >= 0.5 else DN for v in df.CLV.fillna(0.5)]), 2, 1)
  fig.add_hline(y=0.5, line_dash="dash", line_color="#868e96", row=2, col=1)
  fig.add_trace(go.Bar(x=df.index, y=df.UpShadow, name="上影線", showlegend=False,
                       marker_color=DN), 3, 1)
  fig.add_trace(go.Bar(x=df.index, y=-df.LowShadow, name="下影線", showlegend=False,
                       marker_color=UP), 3, 1)
  fig.add_trace(go.Bar(x=df.index, y=df.VolRatio, name="量比", showlegend=False,
                       marker_color=["#b197fc" if v >= 1.8 else "#868e96" for v in df.VolRatio.fillna(0)]), 4, 1)
  fig.add_hline(y=1, line_dash="dash", line_color="#868e96", row=4, col=1)
  fig.add_trace(go.Scatter(x=df.index, y=df.BBW, name="帶寬", showlegend=False,
                           line=dict(color="#ced4da", width=1.2)), 5, 1)
  sq = df[df.Squeeze]
  fig.add_trace(go.Scatter(x=sq.index, y=sq.BBW, mode="markers", name="壓縮", showlegend=False,
                           marker=dict(color="#ffa94d", size=6)), 5, 1)
  _fx_layout(fig, 1200)
  fig.update_layout(barmode="relative", margin=dict(t=110, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0))
  return fig


def _fx_render_wyckoff(link: pd.DataFrame, name: str):
  wk = fx_add_wyckoff(link)
  if wk.Phase.iloc[-1] == "":
    st.warning("資料不足以判定威科夫階段（需約 80 個交易日），請換長一點的期間。")
  phase_now = wk.Phase.iloc[-1] or "資料不足"
  phase_days = int((wk.Phase[::-1] == wk.Phase.iloc[-1]).cumprod().sum())
  checks = fx_wk_checklist(wk)
  passed = sum(ok for _, ok, _ in checks)
  last_vr = wk.VolRatio.dropna()

  k1, k2, k3, k4 = st.columns(4)
  k1.metric("目前階段", phase_now, f"已持續 {phase_days} 天", delta_color="off", border=True)
  k2.metric("收盤位置 CLV", f"{wk.CLV.iloc[-1]:.2f}",
            help="(收-低)/(高-低)；收盤＝主力表態，>0.5 買方掌控", border=True)
  k3.metric("量比", f"{last_vr.iloc[-1]:.2f}" if len(last_vr) else "—",
            help="成交量 ÷ 20 日均量；盤中量為 0 時取前一日", border=True)
  k4.metric("多方轉折條件", f"{passed} / {len(checks)}", border=True)

  left, right = st.columns(2)
  with left.container(border=True):
    st.markdown("**🔄 多方轉折檢查表**")
    for label, ok, note in checks:
      st.markdown(f"{'✅' if ok else '⬜'} **{label}**  \n<small>{note}</small>", unsafe_allow_html=True)
  with right.container(border=True):
    st.markdown("**🧭 開高低收量的分工**")
    st.markdown(
      "- **開盤**＝隔夜情緒（跳空）\n"
      "- **最高／最低**＝掃停損、吃流動性（影線）\n"
      "- **收盤**＝主力表態（CLV）\n"
      "- **成交量**＝力道真假（量比、量價背離）\n"
      "- **帶寬壓縮**＝非線性變盤前兆\n\n"
      "階段：🟦吸籌 → 🟥拉抬 → 🟧出貨 → 🟩收割")

  st.plotly_chart(fx_wyckoff_figure(wk, name))
  st.markdown("**📌 近 40 個交易日事件**")
  ev = fx_wk_events(wk)
  if ev.empty:
    st.caption("近期沒有偵測到事件。")
  else:
    st.dataframe(ev, width="stretch", hide_index=True)
  st.caption("階段判定為簡化規則（均線排列＋MA60 斜率），事件為量化近似，"
             "僅供理解主力行為的框架，不構成投資建議。")


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

  t1, t2, t5, t3, t4 = st.tabs(["💵 匯率技術分析", "📈 股匯連動", "🎯 主力痕跡・威科夫",
                                "🔍 報酬散佈圖", "🗂️ 資料表"])

  with t5:
    _fx_render_wyckoff(link, name)

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
