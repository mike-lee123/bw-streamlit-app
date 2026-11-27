"""第 26 艙「⏱️ 盤中監控」：盤中判斷持股該賣出／續抱、候選股能不能買。

規則（固定、不接 AI；訊號是規則的機械輸出，未經回測驗證，不構成投資建議）：
  持股
    🔴 立即賣出      盤中跌破停損（成本 − 2×ATR14）。硬停損，任何時段都成立
    🟠 警戒          盤中跌破 13MA（緩衝 0.3%，連續 2 次檢查才算）；先提醒、不動作
    🔴 收盤前賣出    13:20 之後仍在 13MA 下方（＝日線「收盤跌破 13MA」）
    🟡 轉強可留      警戒後站回 13MA 且站上當日均價（VWAP）
    🟡 可賣一半      獲利達 2R（R = 2×ATR14）
    🟢 續抱          其他
  候選（前一天報告的買進名單＋自選觀察）
    ⚪ 放棄          開盤高於昨收 2% 以上（不追高，當天不再考慮）
    ⏳ 等待          09:30 前
    ⛔ 暫停買進      大盤濾網轉空（台指期或加權指數跌幅 ≥ 1.5%）
    🟢 可買進（強勢）站上開盤價與 VWAP，且預估量比 ≥ 1.5
    🟢 可買進（拉回）在 13MA 上方 1% 內止穩，且站回 VWAP
    ⚪ 觀望          其他
  時段：09:00–09:15 只看硬停損；13:25 之後（試撮）與收盤後不發新訊號。
  13MA 用「前 12 日收盤 + 即時價」算即時 13MA，與收盤判斷一致。

資料：左側控制台以正式帳號登入永豐 → Shioaji snapshots（即時）；
  沒登入或模擬帳號（模擬環境沒有行情）→ 證交所基本市況報導（延遲約 5 秒，均價取自 Yahoo 1 分 K）；
  都取不到 → Yahoo 1 分 K（延遲約 20 分鐘）。
提醒：每檔記住目前狀態，只有狀態改變才跳通知，避免價格在 13MA 上下來回時一直提醒。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

TZ = dt.timezone(dt.timedelta(hours=8))
BUFFER = 0.003            # 跌破 13MA 要超過 0.3% 才算
CONFIRM_CHECKS = 2        # 連續幾次檢查都在 13MA 下才進入警戒
CHASE = 0.02              # 開盤高於昨收 2% 不追
MARKET_STOP = -1.5        # 大盤跌幅 ≥ 1.5% 暫停買進
VOL_RATIO_MIN = 1.5
PULLBACK_BAND = 0.01      # 在 13MA 上方 1% 內算拉回
ATR_MULT, TP_R = 2.0, 2.0


# ---------- 時段 ----------
def phase_of(now: dt.datetime) -> str:
  t = now.astimezone(TZ).time()
  if now.astimezone(TZ).weekday() >= 5:
    return "休市"
  if t < dt.time(9, 0):
    return "盤前"
  if t < dt.time(9, 15):
    return "開盤震盪"
  if t < dt.time(9, 30):
    return "早盤"
  if t < dt.time(13, 20):
    return "盤中"
  if t < dt.time(13, 25):
    return "收盤確認"
  if t < dt.time(13, 30):
    return "試撮"
  return "收盤後"


def elapsed_fraction(now: dt.datetime) -> float:
  """已開盤時間佔全天 270 分鐘的比例（至少 15 分鐘，避免開盤時量比失真）"""
  n = now.astimezone(TZ)
  mins = (n.hour * 60 + n.minute) - 9 * 60
  return min(1.0, max(15, mins) / 270)


# ---------- 參考值（前一個完整交易日為止） ----------
@st.cache_data(ttl=1800, show_spinner=False)
def daily_hist(codes: tuple, _fetch) -> dict:
  """每檔的完整日 K（已用官方資料補齊落後的日子）"""
  out = {}
  for code in codes:
    df = _fetch(code, range_str="6mo")
    if df is not None and len(df) >= 30:
      out[code] = df
  return out


def ref_for(df: pd.DataFrame, session: dt.date) -> dict | None:
  """以「報價所屬交易日」之前的完整日 K 計算參考值：前 12 日收盤合計（即時 13MA）、ATR14、20 日均量、昨收。
  盤前看到的是昨天的報價，參考值就要用前天以前的資料，才不會把同一天算兩次。"""
  d = df[df.index.date < session]
  if len(d) < 30:
    return None
  c = d["Close"]
  tr = pd.concat([d.High - d.Low, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1)
  return {"last12": float(c.tail(12).sum()), "atr": float(tr.rolling(14).mean().iloc[-1]),
          "vol20": float(d.Volume.tail(20).mean()), "prev_close": float(c.iloc[-1])}


# ---------- 即時報價 ----------
SJ_TIMEOUT_MS = 5000      # 永豐 snapshots 逾時（預設 30 秒；連不上時每輪會卡很久，表格一直出不來）
SJ_COOLDOWN = 600         # 永豐失敗後，這段秒數內直接用 Yahoo，不再每輪重試


def _contract(api, code: str, resolve):
  """查到的合約記在 session（查合約可能觸發重新下載合約表，不能每輪都查）；查不到不記，下次再查"""
  cache = st.session_state.setdefault("im_contracts", {})
  key = (id(api), code)
  if cache.get(key) is None:
    try:
      cache[key] = resolve(api, code)
    except Exception:  # noqa: BLE001
      return None
  return cache[key]


def _contracts_status(api) -> str:
  return str(getattr(getattr(api, "Contracts", None), "status", "")).split(".")[-1] or "未知"


def sj_usable() -> bool:
  return dt.datetime.now(TZ).timestamp() >= st.session_state.get("im_sj_off_until", 0)


def sj_failed(err: str) -> None:
  st.session_state["im_sj_off_until"] = dt.datetime.now(TZ).timestamp() + SJ_COOLDOWN
  st.session_state["im_sj_err"] = err


def quotes_shioaji(api, codes: list[str], resolve) -> dict:
  contracts, ok = [], []
  for c in codes:
    k = _contract(api, c, resolve)
    if k is not None:
      contracts.append(k)
      ok.append(c)
  if not contracts:
    raise RuntimeError(f"查不到任何代號的合約（合約表狀態：{_contracts_status(api)}）")
  snaps = api.snapshots(contracts, timeout=SJ_TIMEOUT_MS)
  if not snaps:
    raise RuntimeError(f"合約有 {len(contracts)} 檔，但永豐快照回傳 0 筆（模擬環境沒有行情，或行情連線未建立）")
  out = {}
  for code, s in zip(ok, snaps):
    last = float(s.close)
    out[code] = {"last": last, "open": float(s.open), "high": float(s.high), "low": float(s.low),
                 "vwap": float(s.average_price) or last, "cum_vol": float(s.total_volume) * 1000,   # 張 → 股
                 "prev_close": last - float(s.change_price),
                 "ts": pd.Timestamp(s.ts, unit="ns").strftime("%H:%M:%S"), "src": "永豐即時",
                 "date": pd.Timestamp(s.ts, unit="ns").date()}
  return out


def _yahoo_1m(symbol: str) -> dict | None:
  r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                   params={"range": "1d", "interval": "1m"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
  if r.status_code != 200:
    return None
  res = (r.json().get("chart", {}).get("result") or [None])[0]
  if not res or not res.get("timestamp"):
    return None
  q = res["indicators"]["quote"][0]
  d = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                   index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ)).dropna(subset=["close"])
  if d.empty:
    return None
  vol = d["volume"].fillna(0)
  tp = (d["high"] + d["low"] + d["close"]) / 3
  vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else float(d["close"].iloc[-1])
  meta = res.get("meta", {})
  return {"last": float(d["close"].iloc[-1]), "open": float(d["open"].iloc[0]), "high": float(d["high"].max()),
          "low": float(d["low"].min()), "vwap": vwap, "cum_vol": float(vol.sum()),
          "prev_close": float(meta.get("chartPreviousClose") or meta.get("previousClose") or np.nan),
          "ts": d.index[-1].strftime("%m/%d %H:%M"), "src": "Yahoo 延遲", "date": d.index[-1].date()}


def _mis(ex_ch: list[str]) -> list[dict]:
  r = requests.get("https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                   params={"ex_ch": "|".join(ex_ch), "json": "1", "delay": "0",
                           "_": int(dt.datetime.now().timestamp() * 1000)},
                   headers={"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}, timeout=8)
  return [m for m in (r.json().get("msgArray") or []) if m.get("c")]


def _f(x) -> float:
  try:
    return float(x)
  except (TypeError, ValueError):
    return float("nan")


def quotes_mis(codes: list[str]) -> dict:
  """證交所「基本市況報導」即時報價（延遲約 5 秒，免帳號）。不知道上市或上櫃，所以兩邊都查。
  它不提供成交均價：VWAP 改用 Yahoo 1 分 K 估（延遲），取不到就用 (高+低+現價)/3。"""
  out = {}
  for i in range(0, len(codes), 20):                       # 一次最多查 20 檔（40 個 ex_ch）
    part = codes[i:i + 20]
    for m in _mis([f"{ex}_{c.lower()}.tw" for c in part for ex in ("tse", "otc")]):
      last = _f(m.get("z"))
      if last != last:                                     # 這 5 秒沒成交：用最佳買價
        last = _f(str(m.get("b") or "").split("_")[0])
      if last != last:
        continue
      t = pd.Timestamp(int(m["tlong"]), unit="ms", tz="UTC").tz_convert(TZ)
      hi, lo = _f(m.get("h")), _f(m.get("l"))
      out[m["c"]] = {"last": last, "open": _f(m.get("o")), "high": hi, "low": lo,
                     "vwap": float(np.nanmean([hi, lo, last])), "vwap_src": "估算",
                     "cum_vol": _f(m.get("v")) * 1000, "prev_close": _f(m.get("y")),
                     "ts": t.strftime("%H:%M:%S"), "src": "證交所即時", "date": t.date()}
  for c, q in out.items():                                 # 均價：Yahoo 1 分 K 的 VWAP（延遲）
    for suf in (".TW", ".TWO"):
      try:
        y = _yahoo_1m(c + suf)
      except Exception:  # noqa: BLE001
        y = None
      if y and y["date"] == q["date"]:
        q["vwap"], q["vwap_src"] = y["vwap"], "Yahoo"
        break
  return out


def quotes_yahoo(codes: list[str]) -> dict:
  out = {}
  for c in codes:
    for suf in (".TW", ".TWO"):
      try:
        q = _yahoo_1m(c + suf)
      except Exception:  # noqa: BLE001
        q = None
      if q:
        out[c] = q
        break
  return out


def market_change(api, resolve) -> tuple[float | None, str]:
  """大盤濾網：有永豐用台指期近月，否則用 Yahoo 加權指數"""
  if api is not None and sj_usable():
    k = _contract(api, "TXFR1", resolve)
    if k is not None:
      try:
        s = api.snapshots([k], timeout=SJ_TIMEOUT_MS)[0]
        return float(s.change_rate), "台指期近月（永豐）"
      except Exception:  # noqa: BLE001
        pass
  try:                                        # 證交所即時加權指數（延遲約 5 秒）
    m = _mis(["tse_t00.tw"])
    if m and _f(m[0].get("z")) == _f(m[0].get("z")) and _f(m[0].get("y")) > 0:
      return (_f(m[0]["z"]) / _f(m[0]["y"]) - 1) * 100, "加權指數（證交所即時）"
  except Exception:  # noqa: BLE001
    pass
  try:
    q = _yahoo_1m("%5ETWII")
    if q:
      from fx_cabin import fx_fetch             # 已用證交所資料補齊的加權指數日線
      idx = fx_fetch("^TWII", "1mo")
      prev = idx[idx.index.date < q["date"]]["Close"]
      if len(prev):                           # 不用 Yahoo meta 的昨收（常常過期）
        return (q["last"] / float(prev.iloc[-1]) - 1) * 100, "加權指數（Yahoo 延遲）"
  except Exception:  # noqa: BLE001
    pass
  return None, "取不到"


# ---------- 規則 ----------
def eval_holding(q: dict, ref: dict, cost: float | None, prev: dict, phase: str) -> tuple[str, str, dict]:
  """回傳 (訊號, 理由, 新狀態)。prev 是這檔上一次的狀態（below_cnt、signal）。"""
  last, vwap = q["last"], q["vwap"]
  ma13 = (ref["last12"] + last) / 13
  R = ATR_MULT * ref["atr"]
  stop = cost - R if cost and cost == cost else None
  below = last < ma13 * (1 - BUFFER)
  cnt = prev.get("below_cnt", 0) + 1 if below else 0
  st_ = {"below_cnt": cnt, "ma13": ma13, "stop": stop}
  was_alert = prev.get("signal", "").startswith(("🟠", "🔴 收盤前"))
  if stop is not None and last < stop:
    return "🔴 立即賣出", f"跌破停損 {stop:,.2f}（成本 − 2×ATR14），硬停損", st_
  if phase == "開盤震盪":
    return "🟢 續抱", f"開盤震盪時段只看硬停損（即時 13MA {ma13:,.2f}）", st_
  if below and phase in ("收盤確認", "試撮"):
    return "🔴 收盤前賣出", f"13:20 後仍在即時 13MA {ma13:,.2f} 下方（{(last / ma13 - 1) * 100:+.2f}%）", st_
  if below and cnt >= CONFIRM_CHECKS:
    return "🟠 警戒", f"盤中跌破即時 13MA {ma13:,.2f}（{(last / ma13 - 1) * 100:+.2f}%）；13:20 仍在下方則收盤前賣出", st_
  if cost and cost == cost and last >= cost + TP_R * R:
    return "🟡 可賣一半", f"獲利 {(last / cost - 1) * 100:+.1f}% 已達 2R（{cost + TP_R * R:,.2f}），其餘以 13MA 移動停利", st_
  if was_alert and last >= ma13 and last >= vwap:
    return "🟡 轉強可留", f"站回即時 13MA {ma13:,.2f} 與均價 {vwap:,.2f}", st_
  return "🟢 續抱", f"即時 13MA {ma13:,.2f}（{(last / ma13 - 1) * 100:+.2f}%）、均價 {vwap:,.2f}", st_


def eval_candidate(q: dict, ref: dict, prev: dict, phase: str, mkt: float | None, now: dt.datetime) -> tuple[str, str, dict]:
  last, vwap, op = q["last"], q["vwap"], q["open"]
  pc = ref["prev_close"]
  ma13 = (ref["last12"] + last) / 13
  st_ = {"abandon": prev.get("abandon", False), "ma13": ma13}
  gap = (op / pc - 1) * 100 if pc else 0.0
  if st_["abandon"] or (phase not in ("盤前", "休市") and op > pc * (1 + CHASE)):
    st_["abandon"] = True
    return "⚪ 放棄", f"開盤 {op:,.2f} 高於昨收 {gap:+.2f}%（> 2% 不追高），今天不再考慮", st_
  if phase in ("盤前", "開盤震盪", "早盤"):
    return "⏳ 等待", "09:30 後才判斷買點", st_
  if phase in ("試撮", "收盤後", "休市"):
    return "⚪ 觀望", "試撮與收盤後不發新買訊", st_
  if mkt is not None and mkt <= MARKET_STOP:
    return "⛔ 暫停買進", f"大盤跌幅 {mkt:+.2f}%（≤ {MARKET_STOP}%）", st_
  vr = q["cum_vol"] / (ref["vol20"] * elapsed_fraction(now)) if ref["vol20"] else 0.0
  if last > op and last > vwap and vr >= VOL_RATIO_MIN:
    return "🟢 可買進（強勢）", f"站上開盤 {op:,.2f} 與均價 {vwap:,.2f}，預估量比 {vr:.2f}", st_
  if ma13 <= last <= ma13 * (1 + PULLBACK_BAND) and last >= vwap:
    return "🟢 可買進（拉回）", f"在即時 13MA {ma13:,.2f} 上方 1% 內止穩並站回均價 {vwap:,.2f}", st_
  return "⚪ 觀望", f"未達買點（均價 {vwap:,.2f}、即時 13MA {ma13:,.2f}、預估量比 {vr:.2f}）", st_


# ---------- 介面 ----------
def render_monitor(deps: dict, holdings: pd.DataFrame, candidates: list[str], names: dict):
  """deps：fetch, sj（回傳 (api 或 None)）, resolve, sj_retry（fn → 結果）"""
  st.caption("盤中依固定規則判斷持股賣出／續抱、候選股能否買進；只有訊號改變時才跳通知。"
             "訊號是規則的機械輸出，未經回測驗證，不構成投資建議。網頁要保持開著才會持續檢查。")
  c1, c2, c3 = st.columns([1, 1, 2])
  on = c1.toggle("啟動監控", key="im_on")
  every = c2.selectbox("檢查間隔", [60, 120, 300], format_func=lambda s: f"{s // 60} 分鐘", key="im_every")
  extra = c3.text_input("額外觀察代號（逗號分隔，當候選買點監看）", key="im_extra")
  cands = list(dict.fromkeys(candidates + [x.strip().upper() for x in extra.split(",") if x.strip()]))
  hold_codes = list(holdings["代號"]) if not holdings.empty else []
  if not hold_codes and not cands:
    st.info("沒有要監控的股票：請在「💼 我的持股」填持股，或在上面輸入觀察代號。")
    return

  def tick():
    now = dt.datetime.now(TZ)
    phase = phase_of(now)
    codes = list(dict.fromkeys(hold_codes + cands))
    with st.spinner("抓取報價中…"):
      hist = daily_hist(tuple(codes), deps["fetch"])
      api = deps["sj"]()
      sim = api is not None and deps.get("sim", lambda: False)()   # 已登入、但是模擬帳號
      if sim:                                     # 模擬帳號沒有行情：不查永豐，直接用證交所即時報價
        api = None
      quotes = {}
      if api is not None and sj_usable():
        try:
          # 連線過期（SessionNotEstablished）時 sj_retry 會重新登入一次再試
          quotes = deps["sj_retry"](lambda a: quotes_shioaji(a, codes, deps["resolve"]))
        except RuntimeError as e:                   # 本模組自己丟出的診斷訊息，內容可控
          sj_failed(str(e))
        except Exception as e:  # noqa: BLE001
          sj_failed(type(e).__name__)
      if len(quotes) < len(codes):
        try:
          quotes.update(quotes_mis([c for c in codes if c not in quotes]))
        except Exception:  # noqa: BLE001
          pass
      if len(quotes) < len(codes):
        quotes.update(quotes_yahoo([c for c in codes if c not in quotes]))
      mkt, mkt_src = market_change(api, deps["resolve"])
    sj_live = any(q.get("src") == "永豐即時" for q in quotes.values())
    if api is not None and not sj_live and st.session_state.get("im_sj_err"):
      left = max(1, (int(st.session_state.get("im_sj_off_until", 0) - now.timestamp()) + 59) // 60)
      st.warning(f"永豐即時報價取不到（{st.session_state['im_sj_err']}），先改用證交所公開報價，"
                 f"約 {left} 分鐘後自動再試永豐。")

    state = st.session_state.setdefault("im_state", {})
    alerts = st.session_state.setdefault("im_alerts", [])
    if state.get("_day") != now.date().isoformat():          # 換日就清狀態
      state.clear()
      state["_day"] = now.date().isoformat()
    rows = []
    for role, lst in (("持股", hold_codes), ("候選", cands)):
      for code in lst:
        q = quotes.get(code)
        ref = ref_for(hist[code], q["date"]) if q and code in hist else None
        if q and ref:
          q = {**q, "prev_close": ref["prev_close"]}   # 漲跌以官方補齊後的昨收為準
        if not q or not ref:
          rows.append({"角色": role, "代號": code, "名稱": names.get(code, ""), "訊號": "⚪ 無資料",
                       "理由": "抓不到即時報價或日 K"})
          continue
        key = f"{role}:{code}"
        prev = state.get(key, {})
        if role == "持股":
          cost = holdings.set_index("代號").loc[code, "成本"]
          sig, why, new = eval_holding(q, ref, None if pd.isna(cost) else float(cost), prev, phase)
        else:
          sig, why, new = eval_candidate(q, ref, prev, phase, mkt, now)
        new["signal"] = sig
        if prev.get("signal") and prev["signal"] != sig and phase not in ("收盤後", "休市", "盤前"):
          msg = f"{code} {names.get(code, '')}：{prev['signal']} → {sig}"
          alerts.insert(0, {"時間": now.strftime("%H:%M:%S"), "代號": code, "變化": f"{prev['signal']} → {sig}", "理由": why})
          st.toast(msg, icon="🔔")
        state[key] = new
        rows.append({"角色": role, "代號": code, "名稱": names.get(code, ""), "訊號": sig,
                     "現價": round(q["last"], 2), "漲跌%": round((q["last"] / q["prev_close"] - 1) * 100, 2)
                     if q["prev_close"] == q["prev_close"] else None,
                     "即時13MA": round(new["ma13"], 2), "均價VWAP": round(q["vwap"], 2), "理由": why,
                     "報價": f"{q['src']} {q['ts']}"})

    m1, m2, m3 = st.columns(3)
    m1.metric("時段", phase, now.strftime("%H:%M:%S"), delta_color="off")
    m2.metric("大盤濾網", f"{mkt:+.2f}%" if mkt is not None else "—", mkt_src, delta_color="off")
    srcs = {q.get("src") for q in quotes.values()}
    src = "永豐即時" if sj_live else "證交所即時（約 5 秒）" if "證交所即時" in srcs else "Yahoo 延遲（約 20 分鐘）"
    m3.metric("報價來源", src,
              "模擬帳號無行情，改用證交所" if sim else ("永豐連線失敗，暫用公開報價" if api is not None and not sj_live else None),
              delta_color="off")
    if "證交所即時" in srcs:
      st.caption("證交所即時報價不含成交均價：均價（VWAP）取自 Yahoo 1 分 K（延遲約 20 分鐘），取不到時以（高＋低＋現價）÷ 3 估算。")
    if phase in ("盤前", "收盤後", "休市"):
      st.info(f"目前為{phase}：表格顯示最近一個交易時段的最後狀態，不發新提醒。")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if alerts:
      st.markdown("**🔔 今日訊號變化**")
      st.dataframe(pd.DataFrame(alerts[:50]), hide_index=True, width="stretch")

  if not sj_usable() and st.button("🔁 立即重試永豐（改好設定、重新連線後按）", key="im_sj_retry"):
    st.session_state["im_sj_off_until"] = 0
    st.session_state.pop("im_sj_err", None)
    st.session_state.pop("im_contracts", None)
  st.fragment(run_every=every if on else None)(tick)()
