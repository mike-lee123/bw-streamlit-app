# 併武 V18.83 dashboard FullMaster

以 `bingwu-stock-dashboard` 的 V18.82（`app.py`）為底，新增第 22 艙「📐 13MA的魔力艙」：走勢圖、開盤劇情、回測、閉環系統、零股隔日沖、費氏座標。13MA 的程式與台股代號表都併在 `app.py` 裡，其餘 21 艙原封不動。

不設密碼，打開網址就能使用。教學與研究用途，不構成投資建議。

## 檔案

| 檔案／資料夾 | 用途 |
|---|---|
| `app.py` | 主程式（22 個艙） |
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
