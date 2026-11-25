"""臺指選擇權（TXO）Put/Call Ratio：期交所每日公布的真實數據。

來源：https://www.taifex.com.tw/cht/3/pcRatio（下載 CSV：pcRatioDown，Big5，一次最多查一個月）。
期交所連續快速請求時偶爾逾時，所以分月抓、每次間隔、失敗重試。
給「⚔️ 選擇權」艙顯示，以及「📋 每日綜合戰情總結艙」投票用。

判讀（台股常見看法）：未平倉 PCR = 賣權未平倉 ÷ 買權未平倉。
賣 Put 的多半是法人／主力（在下方提供支撐），所以未平倉 PCR 高＝支撐強、偏多；低＝支撐弱、偏空。
"""

from __future__ import annotations

import datetime as dt
import io
import time

import pandas as pd
import requests
import streamlit as st

PCR_URL = "https://www.taifex.com.tw/cht/3/pcRatioDown"
PCR_BULL, PCR_BEAR = 100.0, 80.0      # 未平倉 PCR ≥ 100 偏多、≤ 80 偏空
COLS = {
  "賣權成交量": "put_vol", "買權成交量": "call_vol", "買賣權成交量比率%": "vol_pcr",
  "賣權未平倉量": "put_oi", "買權未平倉量": "call_oi", "買賣權未平倉量比率%": "oi_pcr",
}


def _fetch_month(start: dt.date, end: dt.date) -> pd.DataFrame:
  last_err = None
  for wait in (0, 2, 5):
    time.sleep(wait)
    try:
      r = requests.post(PCR_URL, timeout=(5, 20), headers={"User-Agent": "Mozilla/5.0"},
                        data={"down_type": "1", "queryStartDate": start.strftime("%Y/%m/%d"),
                              "queryEndDate": end.strftime("%Y/%m/%d")})
      r.raise_for_status()
      text = r.content.decode("big5", errors="replace")
      if not text.startswith("日期"):
        raise ValueError("期交所回傳的不是 CSV（可能是查詢日期錯誤）")
      return pd.read_csv(io.StringIO(text), index_col=False)
    except Exception as e:  # noqa: BLE001
      last_err = e
  raise RuntimeError(f"期交所 PCR 抓取失敗：{last_err}")


@st.cache_data(ttl=3600, show_spinner="下載期交所選擇權 PCR…")
def taifex_pcr(months: int = 4) -> pd.DataFrame:
  """最近 months 個月的每日 PCR。index＝日期；欄位見 COLS 的英文名。"""
  end = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
  parts = []
  for i in range(months):
    start = end - dt.timedelta(days=29)
    try:
      parts.append(_fetch_month(start, end))
    except Exception:
      if i == 0:          # 最近一個月都抓不到 → 視為失敗；較舊的月份抓不到就少畫一段
        raise
      break
    end = start - dt.timedelta(days=1)
    time.sleep(0.5)
  df = pd.concat(parts).rename(columns=COLS)
  df["日期"] = pd.to_datetime(df["日期"])
  df = df.dropna(subset=["oi_pcr"]).drop_duplicates("日期").set_index("日期").sort_index()
  return df[list(COLS.values())].astype(float)


def pcr_signal(oi_pcr: float) -> tuple[int, str]:
  if oi_pcr >= PCR_BULL:
    return 1, "支撐強・偏多"
  if oi_pcr <= PCR_BEAR:
    return -1, "支撐弱・偏空"
  return 0, "中性"
