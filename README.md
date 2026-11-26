# 併武 V18.83 dashboard FullMaster

以 `bingwu-stock-dashboard` 的 V18.82（`app.py`）為底，新增：

- 第 22 艙「📐 13MA的魔力艙」：走勢圖、開盤劇情、回測、閉環系統、零股隔日沖、費氏座標。
- 第 23 艙「📡 法說會籌碼雷達艙」（`radar_cabin.py`）。
- 第 24 艙「⚖️ 期現對照・13MA艙」：台指期 × 加權指數的期現多空、正逆價差、夜盤→開盤、現貨定方向回測。
- 第 25 艙「💱 台幣匯率×台股艙」（`fx_cabin.py`）：USD/TWD 技術分析（K 線、布林、MACD 交叉、RSI）與股匯連動（滾動相關、雙 MACD、報酬散佈圖），以及「🎯 主力痕跡・威科夫」分頁（開高低收量分工、威科夫階段底色、彈簧／上衝回落、背離、多方轉折檢查表），資料來自 Yahoo。網址加 `?cabin=fx` 直接打開。
- 第 26 艙「📋 每日綜合戰情總結艙」（`summary_cabin.py`）：用固定規則（不接 AI）把各艙結論合成一頁報告：8 票大盤燈號（含期交所選擇權 PCR） → 全策略共振選股＋13MA 防線＋雷達法人籌碼 → 依燈號決定部位上限與停損 → 明日觀察條件，可下載 Markdown／HTML。網址加 `?cabin=summary` 直接打開。
- 第 27 艙「🕑 盤後定價艙」（`after_hours_cabin.py`）：14:00～14:30 盤後定價交易，分「🎯 戰略分組」（各分組自選股的盤後成交、占盤中比例、收盤價買不到／賣不掉訊號）、「🏆 全市場排行」、「🔍 個股歷史」（上市可看近 5／10／20 日；上櫃只有最新一天）。資料直接抓證交所、櫃買中心公開資料，不需金鑰。網址加 `?cabin=after` 直接打開。

13MA 與期現對照的程式、台股代號表都併在 `app.py` 裡（共用同一份 13MA 函式庫），兩艙的永豐資料都共用左側控制台的 Shioaji 連線。其餘艙原封不動。

**這個 repo 就是最新版**：之後的修改都直接改這裡的檔案，不要再用 `build_v1883.py` 從舊 repo 重建。

不設密碼，打開網址就能使用。教學與研究用途，不構成投資建議。

## 檔案

| 檔案／資料夾 | 用途 |
|---|---|
| `app.py` | 主程式（27 個艙） |
| `radar_cabin.py` | 第 23 艙 法說會籌碼雷達 |
| `fx_cabin.py` | 第 25 艙 台幣匯率×台股 |
| `summary_cabin.py` | 第 26 艙 每日綜合戰情總結 |
| `after_hours_cabin.py` | 第 27 艙 盤後定價 |
| `taifex_pcr.py` | 期交所臺指選擇權 Put/Call Ratio（選擇權艙、總結艙共用） |
| `requirements.txt` | 套件 |
| `0930強勢股/` | 第 19 艙的選股引擎 |
| `股票Agent skills/` | 第 20 艙 AI 投資研究團隊（研究程式、已完成的研報） |
| `.github/workflows/research.yml` | 「☁️ 在雲端啟動研究」用的 GitHub Actions |
| `.gitignore` | 排除 `.env`、金鑰檔、快取 |

## 開新網址（Streamlit Community Cloud）

### 1. 建新的 GitHub repo

1. <https://github.com> → **+** → **New repository**，名稱例如 `bingwu-dashboard-v2`，選 **Private**，按 **Create repository**。
2. 用 GitHub Desktop：File → Add local repository 選這個資料夾 → 會提示不是 repo，按 **create a repository** → **Publish repository**（勾 Keep this code private）。
   - 用網頁上傳也可以，但資料夾多、檔名有中文，GitHub Desktop 比較不會漏。

### 2. 在 Streamlit Cloud 建新 App

1. <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**。
2. **Repository** 選新 repo，**Branch** `main`，**Main file path** `app.py`，**App URL** 自訂新網址。
3. **Advanced settings** → **Python version** 選 **3.12**。
4. 要用「☁️ 在雲端啟動研究」的話，在 **Secrets** 填（不用的話留空即可）：

   ```toml
   GITHUB_REPO = "你的帳號/bingwu-dashboard-v2"
   ```

   沒設 `GITHUB_REPO` 時，研究會送到舊的 `mike-lee123/bingwu-stock-dashboard`，結果也推回舊 repo，新網址看不到。
5. 按 **Deploy**。原本 `bingwu-stock-dashboard` 的網址不受影響。

### 3. 雲端研究（選用）

在新 repo 的 **Settings → Secrets and variables → Actions** 新增 `ANTHROPIC_API_KEY`（依用量計費，建議在 Anthropic console 設花費上限）。GitHub token 的設定方式與舊 repo 相同，但要改成授權新 repo。

## 注意事項

- **13MA 艙的永豐資料共用左側控制台的 Shioaji 連線**（只讀行情、不下單）。要抓分K請在左側**取消勾選「使用模擬環境」**，模擬環境幾乎沒有歷史 K 線。Streamlit Cloud 主機在美國，若永豐限制海外連線就會連不上，改用 Yahoo 即可。
- **網址是永久的**：只要不刪除 App 就一直有效。免費方案一段時間沒人使用會休眠，打開時按「Yes, get this app back up!」等約 30 秒就恢復。
- **0930 艙**第一次產生盤前名單要下載約 20 個交易日的證交所資料（1–2 分鐘），雲端重開機後要重新下載。
- **Yahoo 偶爾限流**，讀取失敗時稍等幾分鐘再試。
- 本機的 `bw_ultimate_dashboard_v18.83_dashboard_FullMaster.py` 與這裡的 `app.py` 內容相同，由 `build_v1883.py` 產生。
