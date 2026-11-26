"""官方日 K 補齊：Yahoo 的台股日 K 偶爾會晚一天（例如 0050、0052 缺最新一天），用證交所／櫃買中心補上。

  latest_trading_day()  證交所加權指數的最新交易日（當作「應該要有的最後一天」）
  patch_daily(code, df) df 最後一天早於最新交易日時，抓官方當月（必要時含上月）日 K 補上缺的日子
上市：證交所 STOCK_DAY（成交股數）；上櫃：櫃買中心 tradingStock（成交張數 × 1000 換成股）。
官方網站有流量限制，所以只在資料落後時才查，並且序列化、間隔 0.4 秒；查不到就維持 Yahoo 原樣。
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import requests
import streamlit as st

_UA = {"User-Agent": "Mozilla/5.0"}
_lock = threading.Lock()
_last_call = [0.0]


def _get(url: str, params: dict) -> dict:
  with _lock:                                   # 序列化並限速，避免被官方網站擋
    wait = 0.4 - (time.time() - _last_call[0])
    if wait > 0:
      time.sleep(wait)
    try:
      return requests.get(url, params=params, headers=_UA, timeout=10).json()
    finally:
      _last_call[0] = time.time()


def _roc(d: str) -> pd.Timestamp:
  y, m, dd = d.split("/")
  return pd.Timestamp(int(y) + 1911, int(m), int(dd))


def _num(x: str) -> float:
  return float(str(x).replace(",", "")) if str(x).replace(",", "").replace(".", "", 1).lstrip("-").isdigit() else float("nan")


@st.cache_data(ttl=1800, show_spinner=False)
def latest_trading_day() -> pd.Timestamp | None:
  now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
  for month in (now.replace(day=1), (now.replace(day=1) - pd.DateOffset(months=1))):
    try:
      j = _get("https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST", {"date": month.strftime("%Y%m01"), "response": "json"})
      if j.get("stat") == "OK" and j.get("data"):
        return _roc(j["data"][-1][0])
    except Exception:  # noqa: BLE001
      continue
  return None


@st.cache_data(ttl=1800, show_spinner=False)
def _official_month(code: str, month: str) -> pd.DataFrame:
  """month = 'YYYY-MM'。先試上市，再試上櫃。"""
  y, m = month.split("-")
  rows = []
  try:
    j = _get("https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
             {"date": f"{y}{m}01", "stockNo": code, "response": "json"})
    if j.get("stat") == "OK":
      rows = [(_roc(r[0]), _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6]), _num(r[1])) for r in j.get("data", [])]
  except Exception:  # noqa: BLE001
    rows = []
  if not rows:
    try:
      j = _get("https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock",
               {"code": code, "date": f"{y}/{m}/01", "response": "json"})
      data = ((j.get("tables") or [{}])[0].get("data")) or []
      rows = [(_roc(r[0]), _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6]), _num(r[1]) * 1000) for r in data]
    except Exception:  # noqa: BLE001
      rows = []
  df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]).set_index("Date")
  return df.dropna(subset=["Close"])


def patch_daily(code: str, df: pd.DataFrame) -> pd.DataFrame:
  """Yahoo 日 K 落後最新交易日時，用官方資料補上缺的日子（只補 df 最後一天之後的日子）。"""
  if df is None or df.empty or not str(code).isalnum():
    return df
  latest = latest_trading_day()
  last = pd.Timestamp(df.index[-1]).normalize()
  if latest is None or last >= latest:
    return df
  months = sorted({last.strftime("%Y-%m"), latest.strftime("%Y-%m")})
  off = pd.concat([_official_month(code, m) for m in months])
  add = off[(off.index > last) & (off.index <= latest)]
  if add.empty:
    return df
  return pd.concat([df, add[df.columns.intersection(add.columns)]]).sort_index()


def fresh_fetch(fetch):
  """包裝原本的 Yahoo 抓取函式：抓完自動用官方資料補齊落後的日子"""
  def _f(code, range_str="1y"):
    return patch_daily(code, fetch(code, range_str=range_str))
  return _f
