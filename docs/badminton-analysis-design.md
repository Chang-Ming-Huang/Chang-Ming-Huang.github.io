# 羽球雙打影像分析系統 — 設計文件

> 本文件是**活文件 (living document)**。核心是一份「需求追溯矩陣」：
> 把「想做的分析行為」一路追到「要偵測的資料 → 可參考的套件/論文 → 已做/未做 → 實驗品質」。
> **每次實驗後請回填「狀態」與「實驗品質」欄位**，並據此調整專案方向。

最後更新：2026-06-02

---

## 1. 專案目標與前提

### 目標
分析**羽球雙打**比賽影片，產出球員行為（移動、陣型、輪轉、擊球）與球路分析，
最終由 LLM 結合羽球知識庫，給出戰術分析與訓練建議。

### 硬體與錄影前提
- **錄影**：手機固定機位（腳架），建議 1080p / 60fps，光線充足。
- **運算**：本地電腦，NVIDIA RTX 4070（12GB VRAM，CUDA 加速）。
- **處理模式**：**離線批次分析**（非即時），可接受較長處理時間、可人工微修。

### 範圍決策（第一階段）
- **不對稱分析**：近端隊（我方 2 人）**精準分析**；遠端隊（對手 2 人）**用球路粗估**。
- **球員區分**：近端兩位採**固定身分（球員 A / B）**，以「開場錨定 + 追蹤 + 回合校正 + 人工微修」達成。
- **機位限制**：低機位手機難拍清楚遠端，故對手不強求個別追蹤。

---

## 2. 系統架構

```
① 影像辨識端 (Perception)
   輸入：影片  →  輸出：逐幀結構化座標（已做球場校正，單位＝公尺）
   - 球員偵測+追蹤 / 球員姿態 / 羽球追蹤 / 球場 Homography 校正
        ↓ 結構化數據
② 資料記錄端 (Data / Storage)
   - L0 逐幀原始 → L1 語意事件（拍/回合）→ L2 文字播報
        ↓ 乾淨事件 + 播報
③ 事後分析端 (Analysis)
   - 規則式計算（動作/陣型/輪轉還原）
   - 羽球雙打知識庫 (RAG)
   - LLM 產生戰術分析與建議
```

### 設計原則
- **確定性計算 vs LLM 推理嚴格分離**：座標、距離、速度、球種、陣型 → 程式算（可驗證）；
  戰術解讀、弱點歸納、訓練建議 → LLM 做。
- **資料格式 (schema) 是三層之間的契約**，先定好再開發。
- **文字播報層 (L2)** 同時解決「人怎麼驗收」與「LLM 怎麼讀」。

---

## 3. 核心需求追溯矩陣 ⭐

> 狀態：✅已做 / 🔶部分 / ⬜未做
> 實驗品質：🟢佳 / 🟡普通 / 🔴差 / ⚪未測（多數項目尚未實作，先記錄論文回報之品質）

### 3.1 影像辨識端（Perception）

| # | 為了做到什麼分析 | 要偵測什麼資料 | 可參考套件/論文 | 狀態 | 實驗品質 | 備註 / 對方向的影響 |
|---|---|---|---|---|---|---|
| P1 | 球員移動（距離/速度/熱區） | 近端球員逐幀位置（球場座標） | YOLOv8/v11 + ByteTrack/BoT-SORT；**muhammadyasin79**（雙打移動分析範本）；RichardPinter | ⬜ | ⚪ | 基礎能力，先做 |
| P2 | 球員固定身分 A/B | 追蹤 ID + 微弱外觀特徵 + 站位 | BoT-SORT；Doubles 論文(2508.13507) | ⬜ | ⚪ | **雙打最難點**；靠回合校正+人工微修 |
| P3 | 角色判斷（前場/後場、左/右） | 兩人相對站位 | 自製規則 | ⬜ | ⚪ | 身分識別安全網 + 戰術用，雙重用途 |
| P4 | 揮拍動作 / 步法 | 球員姿態（骨架關節點） | MediaPipe Pose / MMPose / YOLO-Pose | ⬜ | ⚪ | 近端用；遠端拍不清不做 |
| P5 | 羽球軌跡 | 球的逐幀 (x,y) | **TrackNetV3** | ⬜ | ⚪ | 論文 F1 高；可直接用 |
| P6 | 梯形校正（像素→公尺） | 球場四角點 | OpenCV Homography；Roboflow 球場資料集；RichardPinter | ⬜ | ⚪ | **最先定義**，統一座標系 |

### 3.2 資料記錄端（Data）

| # | 為了做到什麼分析 | 要偵測/計算什麼資料 | 可參考 | 狀態 | 實驗品質 | 備註 |
|---|---|---|---|---|---|---|
| D1 | 擊球瞬間 (hit moment) | 球軌跡轉折（y 低點、x 方向反轉）| **jean50621 CoachAI 模組3（有現成實作）**；MDPI Shot Refinement(2024)；arXiv 2307.16000 / 2306.10293 | ⬜ | ⚪ | **樞紐**，論文回報 F1 90.5%；jean50621 有參考碼 |
| D2 | 單拍 / 回合切分 | 相鄰擊球事件之間 | 自製（建在 D1 上）；**jean50621**；RichardPinter **缺此功能** | ⬜ | ⚪ | 解 RichardPinter 缺口；單雙打通用 |
| D3 | 擊球者判定 | 擊球幀「離球最近的球員」 | **jean50621 CoachAI 模組3（有現成）**；自製 | ⬜ | ⚪ | 依賴 P1+P6+D1 |
| D4 | 球種分類（6 類） | 姿態 + 球軌跡 | **RichardPinter (BST-8)** | 🔶 | ⚪ | 現成可分類「已切好的單拍」 |
| D5 | 落點 | 球軌跡落地點 | TrackNet + 自製規則 | ⬜ | ⚪ | |
| D6 | L0/L1/L2 資料格式 | schema 定義 | 自製（接 RichardPinter 中間結構）| ⬜ | ⚪ | **三層契約，先定好** |
| D7 | 文字播報生成 (L2) | 由 L1 事件組句 | 自製 | ⬜ | ⚪ | 連接數據與 LLM 的橋樑 |

### 3.3 雙打戰術（近端隊精準）

| # | 為了做到什麼分析 | 要偵測/計算什麼資料 | 可參考 | 狀態 | 實驗品質 | 備註 |
|---|---|---|---|---|---|---|
| T1 | 陣型判斷（攻：前後 / 守：左右） | 近端兩人相對位置 | Doubles 論文 | ⬜ | ⚪ | 雙打分析主角 |
| T2 | 輪轉時機與品質 | 陣型隨時間變化 | 自製規則 | ⬜ | ⚪ | 雙打精髓 |
| T3 | 補位 / 中路空檔 | 兩人覆蓋區域 | 自製 | ⬜ | ⚪ | |

### 3.4 對手分析（遠端隊粗估，用球路反推）

| # | 為了做到什麼分析 | 要偵測/計算什麼資料 | 可參考 | 狀態 | 實驗品質 | 備註 |
|---|---|---|---|---|---|---|
| O1 | 對手擊球位置 | 擊球幀時球在對方半場的座標 | 自製（D1 順便產出）| ⬜ | ⚪ | 「球＝對手位置探針」 |
| O2 | 我方把對手逼到哪 | 我方回球落點分布 | 自製關聯 | ⬜ | ⚪ | |
| O3 | 對手從哪反擊 | 對手擊球點分布 | 自製關聯 | ⬜ | ⚪ | |

### 3.5 分析端（Analysis）

| # | 為了做到什麼分析 | 要什麼輸入 | 可參考 | 狀態 | 實驗品質 | 備註 |
|---|---|---|---|---|---|---|
| A1 | 戰術解讀 / 弱點歸納 | L2 播報 + 統計 + 知識庫 | Court to conversation(2025)；LLM + RAG | ⬜ | ⚪ | **專案核心價值**，無成熟開源 |
| A2 | 羽球雙打知識庫 | 戰術原則 / 輪轉 / 補位 | 自建 RAG | ⬜ | ⚪ | |
| A3 | 訓練建議 | A1 + A2 結果 | LLM | ⬜ | ⚪ | 最終產出 |

---

## 4. 資料分層 (Schema 雛形)

> 待 D6 細化。先記錄構想，接 RichardPinter 的中間結構
> `((human_pose, pos, shuttle), video_len, label)`。

- **L0 逐幀原始**：每幀 `{frame, shuttle:(x,y), players:[{id, role, pos:(x,y), pose:[17×(x,y)]}]}`（球場公尺座標）。
- **L1 語意事件**：每一拍 `{rally_id, shot_id, hitter(A/B/對手), hit_pos, landing_pos, shot_type, formation, opponent_est_pos}`。
- **L2 文字播報**：由 L1 自動生成的自然語言敘述（球評風格），供 LLM 閱讀與人工驗收。
- **L3 分析建議**：LLM 輸出的戰術解讀與訓練建議。

---

## 5. 關鍵技術筆記

### 5.1 Hit Detection（擊球瞬間偵測）— 系統樞紐
- **原理**：擊球＝球軌跡轉折；多數擊球發生在球軌跡相對低點、剛要上升處。
- **規則式做法**（建議先做）：
  1. **軌跡平滑**：二次曲線擬合相鄰幀，補 TrackNet 漏接、去雜訊（**最關鍵工程**）。
  2. **峰值法**：(幀, y) 波形找轉折 → 擊球時機。
  3. **方向法**：x 方向反轉 → 換邊/換人擊球。
- **品質參考**：純 TrackNet F1 72.3% → 加平滑與修正後 **F1 90.5%**（BWF 賽事資料）。
- **連鎖效益**：做出後同時解決 D2 單拍切分、D3 擊球者、O1 對手位置反推。

### 5.2 雙打球員固定身分 — 三層遞進（出錯可退化）
```
固定身分 A/B  ←最終目標
   ↑ 出錯往下退
角色/站位（前後場、左右）  ←永遠算得出
   ↑
隊伍（近端 vs 遠端）  ←永遠可靠
```
- **開場錨定**：每局/每發球時人未交錯，指定或自動定「左=A、右=B」。
- **短期追蹤**：ByteTrack/BoT-SORT + **姿態**（比外觀耐遮擋）。
- **微弱外觀差異**：膚色/髮型/護具/鞋色/體型/慣用手，當輔助（非主力；隊友同隊服）。
- **回合邊界校正**：每回合間球死人站定 → 重新指派 ID 回正確身分，**錯誤不累積**。
- **離線人工微修**：標出低信心段落，人工幾秒修正 → 拉到實用級準確。
- **誠實預期**：單一低機位無法零錯誤全自動；採「自動為主 + 回合校正 + 人工微修」。

### 5.3 三大辨識元件 — 角色與硬體成本
辨識層由三個元件分工，**球員「偵測容易、串接難」；羽球「偵測難、串接容易」**，所需工具剛好相反。

| 元件 | 角色 | 看幾幀 | 靠什麼判斷 | 輸出 | GPU 成本 |
|---|---|---|---|---|---|
| **YOLO** | 找人 / 抓骨架 | 單幀 | 物件長相 | 方框 / 關節點 | 🟡 中（可選大小，4070 飛快）|
| **ByteTrack** | 把每幀的人串成同一 ID | — | 動作預測(Kalman)+IoU比對 | 帶 ID 的軌跡 | 🟢 幾乎零（非深度學習，CPU 即可）|
| **TrackNet** | 追羽球 | **連續多幀** | **移動軌跡** | 逐像素熱圖 | 🔴 高（瓶頸）|

- **球員追蹤＝YOLO（找人）＋ ByteTrack（串身分）＋ 球場校正（轉公尺）**，雙打再加「身分校正」鎖定 A/B。
- **羽球只有一顆 → 無身分配對問題**，TrackNet 內部已用多幀把每幀球抓準，串接只要照時間連，故「一個元件就能給球路」。
- **TrackNet 是效能瓶頸**：輸出整張熱圖 + 吃多幀 + V3 修正；約 25 FPS（YOLO 小模型可達數百 FPS）。
  整場處理時間基本由它決定；**60fps 比 30fps 慢一倍**。未來要加速優先優化它（降解析度／跳幀／TensorRT）。
- **分工原則**：人大慢清楚 → YOLO；球小快會糊會消失 → TrackNet（YOLO 追球僅 ~58%，TrackNet ~98%）。

### 5.4 技術名詞速查
- **YOLO (You Only Look Once)**：物件偵測模型，一次掃描整張圖同時「找位置 + 分類別」，快。有 Detection／Pose／Tracking 多種用途；版本 v3→v7→v8→v11 越新越強，用法相近。**偵測人很強，偵測羽球很弱**。
- **YOLO-Pose**：YOLO 的姿態版，抓人體骨架關節點，用於揮拍／步法分析。
- **ByteTrack**：多物件追蹤器。招牌是**連低信心的偵測框也保留**（被遮擋／模糊的人其實是真人），用兩階段比對把跟丟的人撈回，減少 ID 跳號。只靠動作+位置、不看長相，故雙打交錯時仍會跳號。
- **BoT-SORT**：ByteTrack 進階版，多了外觀 Re-ID + 相機運動補償；但雙打同隊服使 Re-ID 助益有限。
- **TrackNet**：專為小快球設計的追蹤模型。看連續多幀、輸出逐像素熱圖，能從軌跡推斷被遮擋／糊掉的球。V3 再加軌跡修正(inpainting)補洞。
- **Homography（單應性矩陣）**：把畫面像素座標換算成真實球場公尺座標的透視校正，用 OpenCV + 球場四角點求得。

---

## 6. 現成資源評估

### 開源專案逐一拆解

| 資源 | 核心功能 | 對本專案價值 |
|---|---|---|
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | **只做羽球追蹤**：軌跡預測 + 軌跡修正(inpainting)。輸出 CSV `Frame,Visibility,X,Y`。品質 F1 98.56%、25 FPS | ✅ **球的地基，直接用**。P5 用它 |
| [jean50621/Badminton_Challenge](https://github.com/jean50621/Badminton_Challenge)（CoachAI）| **最完整，11 模組**：YOLOv7(球員/場/網) + TrackNetv2(球) + **模組3 擊球幀&擊球者** + RTMPose(姿態) + 球種/正反拍/球高/落點/勝負 | ⭐ **唯一內建 hit detection + 擊球者**；D1/D3 的參考實作。架構較重(YAML+檔案串接) |
| [RichardPinter/badminton_shot_type](https://github.com/RichardPinter/badminton_shot_type) | TrackNetV3+MMPose+球場校正(Homography)+球種分類(6類)。中間結構 `((human_pose,pos,shuttle),video_len,label)` | 🔶 **單打用**(m=2＝上下半場各一人)；無回合切分/hit detection/LLM；攔截中間資料當 L0；D4 球種可直接用 |
| [muhammadyasin79/Badminton_Analytics_Project](https://github.com/muhammadyasin79/Badminton_Analytics_Project) | YOLOv8+Pose+自訓YOLO11(球)。**豐富移動分析**：熱區、球場支配度、區域轉移矩陣、回中位置、速度圖。**偏雙打**(Player1/2) | ⭐ **移動/陣型分析範本**(P1/T1/T3)；視覺化可直接借鏡。缺球種/回合切分 |
| [nethra8902/Badminton-Sport-Analysis](https://github.com/nethra8902/Badminton-Sport-Analysis-Computer-Vision) | YOLOv3+MXNet：球員框、球、計時、HIT/出界(AWAY)高亮。單打 | △ 模型較舊；hit 概念陽春，參考價值低 |
| [ToanNguyenKhanh/Badminton-Analysis](https://github.com/ToanNguyenKhanh/Badminton-Analysis) | 純偵測：YOLOv8 框球員 + 羽球 | △ 無追蹤/球場/球種，demo 級 |

### 論文

| 論文 | 用途 | 備註 |
|---|---|---|
| [Doubles 論文(arXiv 2508.13507)](https://arxiv.org/html/2508.13507v1) | 雙打多人追蹤、ID 保持 | 偏研究；姿態法比視覺法轉移性約 2× |
| [MDPI Shot Refinement(2024)](https://www.mdpi.com/1424-8220/24/13/4372) | hit detection + 軌跡平滑 | **必讀**，單目相機，F1 90.5% |
| [Court to conversation(2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125020659) | CV + RAG + LLM 戰術分析 | 與本專案構想最接近；少開源 |
| [BST(arXiv 2502.21085)](https://arxiv.org/pdf/2502.21085) | RichardPinter 用的模型 | 看 m=2 定義 |

### 取用地圖（站在誰的肩膀上）
- 球追蹤 (P5) → **TrackNetV3**（直接用）
- 擊球幀+擊球者 (D1/D3) → **jean50621 CoachAI 模組3**（深入研究、借演算法）
- 球種 (D4) → **RichardPinter (BST)**
- 球場校正 (P6)、姿態 (P4) → RichardPinter / jean50621
- 移動/陣型 (P1/T1/T3) → **muhammadyasin79**（雙打導向）
- L2 播報 + L3 LLM 分析 → **自製（核心價值，無成熟開源）**

---

## 7. 分階段開發計畫

### 階段 0：環境 + 球偵測驗證（從這裡開始）
**為什麼先做球偵測**：① 它是地基（D1/D2/D3/O1 全依賴它）；② 它是效能瓶頸，早跑早知道時間成本；③ 最獨立、成功標準乾淨；④ 順便驗證 GPU 環境。
- **實驗 0**：裝 CUDA + PyTorch GPU + clone TrackNetV3；`torch.cuda.is_available()` 須為 True。
- **實驗 1**：用**自己拍的** 30~60 秒短片（含殺球/吊球/平抽等不同球路）跑 TrackNet，輸出標記影片 + CSV。
  - ⚠️ 要驗證的不是「TrackNet 準不準」（轉播畫面已知 98%），而是 **「在我的手機、低機位、我的球場光線下還剩多少準度」**。開源模型多以職業轉播畫面訓練，換業餘手機影片可能掉，掉多少是未知。
  - 跑通 RichardPinter，觀察其中間輸出格式（順帶驗證 P6/D4）。

**實驗 1 品質檢查表**（看標記影片逐項打勾，結果回填 P5 實驗品質欄）：
- [ ] 漏接率：殺球等高速球有無整段跟丟？
- [ ] 誤判：白線/球衣/燈光有無被當成球？
- [ ] 遮擋表現：球被球員擋住時 V3 有無補回？
- [ ] 過網瞬間：低機位下球過網追得到嗎？
- [ ] 速度：這段幾秒影片跑了多久？→ 推算整場時間
- 判讀：佳→往 D1 走；普通→確認需加軌跡平滑層；差🚩→根本問題（換機位/重訓），越早發現越好

> 註：校正(P6)可晚一步。實驗 1 先在**像素座標**確認「抓得到球」，座標轉公尺是之後的加工。

### 階段 1：近端隊精準分析
- P6 球場校正 → P1 球員追蹤 → P5 球軌跡 → **D1 hit detection** → D2 回合切分 →
  D3 擊球者 → P2/P3 固定身分與角色 → T1/T2 陣型與輪轉。

### 階段 2：對手粗估
- O1 對手擊球位置 → O2/O3 攻防關聯分析。

### 階段 3：分析與建議
- D6/D7 schema 與 L2 播報 → A2 雙打知識庫 → A1/A3 LLM 戰術分析與建議。

---

## 8. 待決事項 / 風險
- [ ] **TrackNet 在手機/低機位畫質下的球偵測品質**（最優先，待實驗 1 回填 P5）— 開源模型多以職業轉播訓練，業餘畫面可能掉。
- [ ] **TrackNet 是效能瓶頸**（~25 FPS）；整場處理時間由它決定，60fps 再翻倍。未來優化首要對象（降解析度/跳幀/TensorRT）。
- [ ] hit detection 軌跡平滑在 60fps 手機畫質下的實際品質（待實驗回填 D1）。
- [ ] 低機位下近端兩人遮擋時 ID 跳號頻率（待實驗回填 P2）。
- [ ] RichardPinter 中間資料能否乾淨攔截、是否需改其碼（待階段 0）。
- [ ] jean50621 CoachAI 模組3（擊球幀/擊球者）能否抽出單獨使用（待研究 D1/D3）。
- [ ] 遠端球路反推的座標誤差是否可接受（待實驗回填 O1）。

---

## 9. 參考文獻
- TrackNetV3: https://github.com/qaz812345/TrackNetV3
- jean50621/Badminton_Challenge (CoachAI, An All Deep System): https://github.com/jean50621/Badminton_Challenge
- RichardPinter/badminton_shot_type: https://github.com/RichardPinter/badminton_shot_type
- muhammadyasin79/Badminton_Analytics_Project: https://github.com/muhammadyasin79/Badminton_Analytics_Project
- Enhancing Badminton Game Analysis (Shot Refinement, MDPI Sensors 2024): https://www.mdpi.com/1424-8220/24/13/4372
- A New Perspective for Shuttlecock Hitting Event Detection (arXiv 2306.10293): https://arxiv.org/html/2306.10293
- Automated Hit-frame Detection (arXiv 2307.16000): https://arxiv.org/pdf/2307.16000
- Bridging the Gap: Doubles Badminton Analysis (arXiv 2508.13507): https://arxiv.org/html/2508.13507v1
- BST: Badminton Stroke-type Transformer (arXiv 2502.21085): https://arxiv.org/pdf/2502.21085
- Court to conversation (RAG-enhanced LLMs, 2025): https://www.sciencedirect.com/science/article/abs/pii/S0950705125020659
