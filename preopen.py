"""開盤前趨勢預測（08:45 台指期開盤、09:00 台股開盤前看）：6 大指標各投 +1／0／−1。

  1. 台積電 ADR（TSM）     美股收盤漲跌 ±1%；另算 ADR 溢價（TSM × 匯率 ÷ 5 對 2330 收盤）
  2. 費城半導體（^SOX）    ±1%
  3. 道瓊指數（^DJI）      ±0.5%
  4. USD/TWD（TWD=X）      對台股前一交易日的變動：台幣升值 ≥ 0.15% +1、貶值 ≥ 0.15% −1
  5. 韓國綜合指數（^KS11） 08:00 開盤後的即時漲跌 ±0.5%（台北 08:00 前尚未開盤，不投票）
  6. 台指期夜盤（期交所）  夜盤收盤對日盤收盤 ±0.5%；夜盤進行中改用永豐 TXFR1 即時價

時效：只採用台股前一交易日收盤「之後」才出現的資料。例如美股休市，前一晚的美股收盤
早就反映在台股前一天的盤勢裡，這一項就不投票（標示「無新資料」）。
合計 ≥ +2 偏多開高、≤ −2 偏空開低，其餘平盤震盪。預估開盤價：台指期 08:45 從夜盤收盤價接續，
加權指數用「前一日收盤 ×（1 + 夜盤漲跌%）」估算（夜盤是期貨，與現貨有價差，僅供參考）。
"""

from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

_UA = {"User-Agent": "Mozilla/5.0"}
TZ = "Asia/Taipei"
BULL_AT, BEAR_AT = 2, -2


def _now() -> pd.Timestamp:
  return pd.Timestamp.now(tz=TZ)


# ---------- 資料 ----------
def _yahoo(symbol: str) -> dict:
  """日線收盤（日期為交易所當地日期）＋最新成交價與時間"""
  j = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                   params={"range": "1mo", "interval": "1d"}, headers=_UA, timeout=10).json()
  r = j["chart"]["result"][0]
  m = r["meta"]
  tz = m.get("exchangeTimezoneName") or "UTC"
  idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
  closes = pd.Series(r["indicators"]["quote"][0]["close"], index=idx, dtype="float64").dropna()
  closes = closes[~closes.index.duplicated(keep="last")]
  last_t = pd.Timestamp(m["regularMarketTime"], unit="s", tz="UTC").tz_convert(tz)
  return {"closes": closes, "last": float(m["regularMarketPrice"]), "last_t": last_t, "tz": tz}


def _taifex_tx(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
  """期交所台指期每日行情（含盤後＝夜盤）。欄位依官方 CSV 順序取用。"""
  r = requests.post("https://www.taifex.com.tw/cht/3/futDataDown",
                    data={"down_type": "1", "commodity_id": "TX",
                          "queryStartDate": start.strftime("%Y/%m/%d"), "queryEndDate": end.strftime("%Y/%m/%d")},
                    headers=_UA, timeout=20)
  df = pd.read_csv(io.StringIO(r.content.decode("big5", errors="replace")), index_col=False, dtype=str)
  df = df.iloc[:, [0, 2, 6, 9, 17]]
  df.columns = ["date", "month", "close", "vol", "session"]
  df["date"] = pd.to_datetime(df["date"], errors="coerce")
  df["month"] = df["month"].str.strip()
  df["close"] = pd.to_numeric(df["close"], errors="coerce")
  df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
  df["night"] = df["session"].str.strip().ne("一般")
  return df.dropna(subset=["date", "close"])


@st.cache_data(ttl=300, show_spinner=False)
def night_session(t_prev: pd.Timestamp, _api=None, _resolve=None, sj_on: bool = False) -> dict | None:
  """前一交易日日盤收盤之後的夜盤：先找期交所已收盤的夜盤，找不到再用永豐即時價（sj_on 只用來區分快取）"""
  api, resolve = _api, _resolve
  try:
    df = _taifex_tx(t_prev - pd.Timedelta(days=10), _now().tz_localize(None).normalize())
    day = df[(df["date"] == t_prev) & ~df["night"] & df["month"].str.fullmatch(r"\d{6}")]
    if not day.empty:
      near = day.sort_values("vol").iloc[-1]                 # 成交量最大的就是近月
      ngt = df[(df["date"] > t_prev) & df["night"] & (df["month"] == near["month"])]
      if not ngt.empty:
        n = ngt.sort_values("date").iloc[0]
        return {"last": float(n["close"]), "day_close": float(near["close"]), "month": near["month"],
                "src": f"期交所夜盤收盤（{near['month']}）", "final": True}
      if api is not None and resolve is not None:
        s = api.snapshots([resolve(api, "TXFR1")])[0]
        ts = pd.Timestamp(int(s.ts), unit="ns", tz="UTC").tz_convert(TZ)
        if ts.tz_localize(None) > t_prev + pd.Timedelta(hours=14):   # 日盤收盤之後的成交
          return {"last": float(s.close), "day_close": float(near["close"]), "month": near["month"],
                  "src": f"永豐 TXFR1 夜盤即時（{ts:%m/%d %H:%M}）", "final": False}
  except Exception:  # noqa: BLE001
    pass
  return None


def _vote(chg: float, th: float) -> int:
  return 1 if chg >= th else -1 if chg <= -th else 0


@st.cache_data(ttl=300, show_spinner="抓取開盤前 6 大指標…")
def forecast(_api=None, _resolve=None, sj_on: bool = False) -> dict:
  """sj_on 只用來區分有無永豐連線的快取"""
  import twse_data
  from fx_cabin import fx_fetch

  now = _now()
  t_prev = twse_data.latest_trading_day()
  if t_prev is None:
    raise RuntimeError("查不到台股最新交易日")
  if now.tz_localize(None).normalize() == t_prev and now.hour >= 9:
    note = "台股今天已開盤，以下是「今天開盤前」的回顧；收盤後要到隔天早上才有新的夜盤與美股資料。"
  else:
    note = ""
  twii = fx_fetch("^TWII", "1mo")["Close"].dropna()
  twii_close = float(twii[twii.index <= t_prev].iloc[-1])
  rows, est = [], {"t_prev": t_prev, "twii_close": twii_close, "note": note, "asof": now.strftime("%m/%d %H:%M")}

  def add(name, val, chg, vote, read, when):
    rows.append({"指標": name, "最新": val, "漲跌%": chg, "票": vote, "讀數": read, "資料時間": when})

  # 1～3 美股：最後一根日線的日期要 ≥ 台股前一交易日才算新資料
  us = {}
  for name, sym, th in (("台積電 ADR（TSM）", "TSM", 1.0), ("費城半導體（SOX）", "^SOX", 1.0), ("道瓊指數（DJI）", "^DJI", 0.5)):
    try:
      q = _yahoo(sym)
      c = q["closes"]
      chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
      fresh = c.index[-1] >= t_prev
      us[sym] = (float(c.iloc[-1]), fresh)
      read = ((f"漲 {chg:.2f}%" if chg > 0 else f"跌 {abs(chg):.2f}%") + f"（門檻 ±{th}%）" if fresh
              else "美股休市，台股前一交易日已反映，無新資料，不投票")
      add(name, float(c.iloc[-1]), chg, _vote(chg, th) if fresh else 0, read, f"美股 {c.index[-1]:%m/%d} 收盤")
    except Exception:  # noqa: BLE001
      add(name, None, None, 0, "取不到資料，不投票", "—")

  # 4 匯率：現價對台股前一交易日 13:30 收盤時的匯價（Yahoo 外匯日線收盤不可靠，改用 60 分 K）
  try:
    q = _yahoo("TWD=X")
    ref = _fx_at(t_prev + pd.Timedelta(hours=13, minutes=30))
    chg = (q["last"] / ref - 1) * 100
    v = 1 if chg <= -0.15 else -1 if chg >= 0.15 else 0
    est["usdtwd"] = q["last"]
    add("USD/TWD", q["last"], chg, v,
        ("台幣升值" if chg < 0 else "台幣貶值") + f" {abs(chg):.2f}%（門檻 ±0.15%，升值偏多）",
        f"{q['last_t'].tz_convert(TZ):%m/%d %H:%M}")
  except Exception:  # noqa: BLE001
    add("USD/TWD", None, None, 0, "取不到資料，不投票", "—")

  # 5 KOSPI：今天（首爾日期）開盤後的即時價才投票
  try:
    q = _yahoo("^KS11")
    c = q["closes"]
    today_kr = now.tz_convert(q["tz"]).tz_localize(None).normalize()
    live = q["last_t"].tz_localize(None).normalize() == today_kr and today_kr > t_prev
    prev = c[c.index < today_kr] if live else c
    if live:
      chg = (q["last"] / float(prev.iloc[-1]) - 1) * 100
      add("韓國綜合指數（KOSPI）", q["last"], chg, _vote(chg, 0.5),
          (f"漲 {chg:.2f}%" if chg > 0 else f"跌 {abs(chg):.2f}%") + "（門檻 ±0.5%）",
          f"即時 {q['last_t'].tz_convert(TZ):%H:%M}（台北）")
    else:
      chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
      add("韓國綜合指數（KOSPI）", float(c.iloc[-1]), chg, 0,
          "今天尚未開盤（台北 08:00 開盤），前一日漲跌已反映，不投票", f"{c.index[-1]:%m/%d} 收盤")
  except Exception:  # noqa: BLE001
    add("韓國綜合指數（KOSPI）", None, None, 0, "取不到資料，不投票", "—")

  # 6 台指期夜盤
  n = night_session(t_prev, _api, _resolve, _api is not None)
  if n:
    pts = n["last"] - n["day_close"]
    chg = pts / n["day_close"] * 100
    est.update(night_last=n["last"], night_pts=pts, night_chg=chg, night_final=n["final"])
    add("台指期夜盤", n["last"], chg, _vote(chg, 0.5),
        f"{pts:+,.0f} 點（日盤收 {n['day_close']:,.0f}；門檻 ±0.5%）", n["src"])
  else:
    add("台指期夜盤", None, None, 0,
        "夜盤尚未收盤（05:00 收盤後期交所才公布；連上永豐可看即時），不投票", "—")

  # ADR 溢價（參考，不投票）
  try:
    tsm, fresh = us["TSM"]
    tw = twse_data.patch_daily("2330", _tw_close("2330.TW"))
    c2330 = float(tw["Close"][tw.index <= t_prev].iloc[-1])
    implied = tsm * est["usdtwd"] / 5
    est.update(c2330=c2330, implied_2330=implied, adr_prem=(implied / c2330 - 1) * 100, adr_fresh=fresh)
  except Exception:  # noqa: BLE001
    pass

  table = pd.DataFrame(rows)
  total = int(table["票"].sum())
  est["total"] = total
  est["view"] = "🔴 偏空開低" if total <= BEAR_AT else "🟢 偏多開高" if total >= BULL_AT else "🟡 平盤震盪"
  if "night_chg" in est:
    est["twii_open"] = twii_close * (1 + est["night_chg"] / 100)
  return {"table": table, "est": est}


def _fx_at(t: pd.Timestamp) -> float:
  """USD/TWD 在台北時間 t 之前最後一根 60 分 K 的收盤"""
  r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/TWD=X",
                   params={"range": "5d", "interval": "60m"}, headers=_UA, timeout=10).json()["chart"]["result"][0]
  s = pd.Series(r["indicators"]["quote"][0]["close"], dtype="float64",
                index=pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(TZ).tz_localize(None)).dropna()
  return float(s[s.index < t].iloc[-1])


def _tw_close(sym: str) -> pd.DataFrame:
  q = _yahoo(sym)
  return q["closes"].to_frame("Close")


# ---------- 報告 ----------
VOTE_ICON = {1: "🔺 +1", 0: "➖ 0", -1: "🔻 −1"}


def show_table(table: pd.DataFrame) -> pd.DataFrame:
  t = table.copy()
  t["最新"] = t["最新"].map(lambda x: "—" if x is None or x != x else f"{x:,.3f}" if x < 100 else f"{x:,.2f}")
  t["漲跌%"] = t["漲跌%"].map(lambda x: "—" if x is None or x != x else f"{x:+.2f}%")
  t["票"] = t["票"].map(VOTE_ICON)
  return t


def section(fc: dict, sells=None, buys=None) -> tuple[str, list]:
  """回傳 (標題, [段落／表格])，可直接放進總結報告"""
  est, table = fc["est"], fc["table"]
  n_vote = int((table["讀數"].str.contains("不投票") == False).sum())  # noqa: E712
  lines = []
  if est["note"]:
    lines.append(f"⚠️ {est['note']}")
  lines.append(f"開盤前判斷：**{est['view']}**（6 項指標中 {n_vote} 項有新資料可投票，合計 {est['total']:+d}；"
               f"≥ +{BULL_AT} 偏多、≤ {BEAR_AT} 偏空）。資料抓取時間 {est['asof']}，"
               f"對照台股前一交易日 {est['t_prev']:%m/%d}。")
  lines.append(show_table(table))
  if "night_last" in est:
    lines.append(f"- **台指期 08:45**：從夜盤{'收盤' if est['night_final'] else '最新'}價 {est['night_last']:,.0f} 接續，"
                 f"較日盤收盤 {est['night_pts']:+,.0f} 點（{est['night_chg']:+.2f}%）。")
    lines.append(f"- **加權指數 09:00**：前一日收盤 {est['twii_close']:,.2f} × 夜盤漲跌 → 估約 **{est['twii_open']:,.0f}**"
                 f"（{est['twii_open'] - est['twii_close']:+,.0f} 點；期現價差會讓實際開盤有落差）。")
  else:
    lines.append(f"- 夜盤資料還沒有，無法估算開盤點位；加權指數前一日收盤 {est['twii_close']:,.2f}。")
  if "adr_prem" in est:
    lines.append(f"- **台積電**：ADR 換算 {est['implied_2330']:,.1f} 元（TSM × USD/TWD {est['usdtwd']:.3f} ÷ 5），"
                 f"對 2330 收盤 {est['c2330']:,.1f} 溢價 {est['adr_prem']:+.1f}%"
                 + ("" if est["adr_fresh"] else "（美股休市，ADR 沒有新價格，溢價只是原本的價差）") + "。"
                 "溢價長期存在，看的是比前一天擴大或縮小。")
  # 和隔日交易訊號對照
  tips = []
  if sells is not None and not sells.empty:
    red = sells[sells["訊號"].astype(str).str.startswith("🔴")]
    if len(red):
      codes = "、".join(red["代號"].astype(str))
      tips.append(f"持股 {codes} 有 🔴 賣出訊號" + ("，開盤偏空，建議開盤後優先執行，不要等反彈。"
                                                if est["total"] <= BEAR_AT else "，依計畫執行。"))
  if buys is not None and not buys.empty:
    tips.append("買進訊號：" + ("開盤偏多容易跳空，超過「開盤上限價」就不追，改等回測 13MA。"
                           if est["total"] >= BULL_AT else
                           "開盤偏空，等回測 13MA 附近再分批，跌破停損價不進場。"
                           if est["total"] <= BEAR_AT else "依「開盤上限價／回測進場價」執行。"))
  if tips:
    lines.append("**對照隔日交易訊號**：" + " ".join(tips))
  lines.append("規則：美股與 ADR、費半 ±1%，道瓊、KOSPI、夜盤 ±0.5%，匯率 ±0.15%；"
               "只採用台股前一交易日收盤後才出現的新資料。預測僅供參考，不構成投資建議。")
  return ("開盤前趨勢預測（08:45 台指期／09:00 台股）", lines)
