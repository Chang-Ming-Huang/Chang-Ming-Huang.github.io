# ============================================================
#  羽球雙打影像分析 — 專案初始化腳本 (Windows / PowerShell)
#  用法：在 PowerShell 視窗執行
#     powershell -ExecutionPolicy Bypass -File .\setup_badminton_project.ps1
#  會在「桌面」建立 badminton-analysis 專案資料夾與骨架。
# ============================================================

$ErrorActionPreference = "Stop"
$proj = "badminton-analysis"
$base = Join-Path ([Environment]::GetFolderPath("Desktop")) $proj

Write-Host ">> 建立專案資料夾：$base" -ForegroundColor Cyan
if (Test-Path $base) {
    Write-Host "!! 資料夾已存在，將沿用既有資料夾（不覆蓋既有檔案）。" -ForegroundColor Yellow
}

# ---- 資料夾結構 ----
$dirs = @(
    "docs",
    "data\raw_videos",          # 放手機錄的原始影片（不進 git）
    "data\processed",           # 處理後資料（不進 git）
    "third_party",              # 之後 clone TrackNetV3 等上游 repo（不進 git）
    "src\perception",           # ① 辨識：YOLO / ByteTrack / TrackNet / 球場校正
    "src\data_layer",           # ② 記錄：L0~L2 schema
    "src\analysis",             # ③ 分析：LLM / RAG / 知識庫
    "experiments\exp01_tracknet",
    "notebooks"
)
foreach ($d in $dirs) {
    $p = Join-Path $base $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

function Write-Utf8($relPath, $content) {
    $full = Join-Path $base $relPath
    if (Test-Path $full) {
        Write-Host "   略過（已存在）：$relPath" -ForegroundColor DarkGray
        return
    }
    $content | Out-File -FilePath $full -Encoding utf8
    Write-Host "   建立：$relPath" -ForegroundColor Green
}

# ---- README ----
Write-Utf8 "README.md" @'
# 羽球雙打影像分析系統

手機錄影 → 本地 GPU(RTX 4070) 離線分析 → LLM 給戰術建議。
完整設計與需求追溯矩陣見 `docs/design.md`。

## 專案結構
- `docs/`        設計文件（活文件，實驗後回填品質）
- `data/`        raw_videos 原始影片 / processed 處理後資料（皆不進 git）
- `third_party/` 上游開源 repo（TrackNetV3 等，不進 git）
- `src/`         perception 辨識 / data_layer 記錄 / analysis 分析
- `experiments/` 各實驗紀錄；先做 exp01_tracknet
- `notebooks/`   探索用

## 第一步（階段 0）
見 `experiments/exp01_tracknet/README.md`：裝環境 → 跑 TrackNet → 填品質檢查表。
'@

# ---- .gitignore ----
Write-Utf8 ".gitignore" @'
# Python
__pycache__/
*.py[cod]
.venv/
venv/
env/
.ipynb_checkpoints/

# 資料與模型權重（大檔，不進 git）
data/raw_videos/
data/processed/
third_party/
*.mp4
*.avi
*.mov
*.mkv
*.pt
*.pth
weights/

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db
'@

# ---- requirements.txt ----
Write-Utf8 "requirements.txt" @'
# 注意：PyTorch(GPU/CUDA 版) 請依官網指令另裝，不要直接 pip install torch
#   https://pytorch.org/get-started/locally/  （選 CUDA 12.x）
# 以下為本專案上層工具；TrackNetV3 等上游 repo 各有自己的 requirements。
ultralytics        # YOLOv8/v11 + ByteTrack/BoT-SORT
opencv-python      # 影像處理、Homography 校正
numpy
pandas
scipy              # 軌跡平滑（二次曲線擬合）
matplotlib         # 熱區圖、軌跡視覺化
'@

# ---- exp01 實驗紀錄（含品質檢查表）----
Write-Utf8 "experiments\exp01_tracknet\README.md" @'
# 實驗 01：TrackNet 球偵測（階段 0）

## 目的
驗證的不是「TrackNet 準不準」（轉播畫面已知 ~98%），
而是「在我的手機、低機位、我的球場光線下，還剩多少準度」。

## 步驟
1. 裝 CUDA + PyTorch GPU 版；確認 `python -c "import torch;print(torch.cuda.is_available())"` 為 True
2. `cd ..\..\third_party` 後 `git clone https://github.com/qaz812345/TrackNetV3`
3. 用自己拍的 30~60 秒短片（含殺球/吊球/平抽）跑 TrackNet，輸出標記影片 + CSV
4. 看標記影片，填下方檢查表

## 品質檢查表（回填設計文件 P5）
- [ ] 漏接率：殺球等高速球有無整段跟丟？
- [ ] 誤判：白線/球衣/燈光有無被當成球？
- [ ] 遮擋表現：球被球員擋住時有無補回？
- [ ] 過網瞬間：低機位下球過網追得到嗎？
- [ ] 速度：幾秒影片跑了多久？→ 推算整場時間

## 判讀
- 佳 → 往 D1 擊球偵測走
- 普通 → 確認需加「軌跡平滑」層
- 差 → 根本問題（換機位/重訓），越早發現越好

## 結果紀錄
（在此填入實際觀察、截圖、時間數據）
'@

# ---- Python 套件佔位 ----
"" | Out-File (Join-Path $base "src\perception\__init__.py") -Encoding utf8 -NoNewline
"" | Out-File (Join-Path $base "src\data_layer\__init__.py") -Encoding utf8 -NoNewline
"" | Out-File (Join-Path $base "src\analysis\__init__.py") -Encoding utf8 -NoNewline
foreach ($k in @("data\raw_videos",
"data\processed","third_party","notebooks")) {
    $gk = Join-Path $base "$k\.gitkeep"
    if (-not (Test-Path $gk)) { "" | Out-File $gk -Encoding utf8 -NoNewline }
}

# ---- 抓最新設計文件 ----
$rawUrl = "https://raw.githubusercontent.com/Chang-Ming-Huang/Chang-Ming-Huang.github.io/claude/badminton-video-analysis-5lDIR/docs/badminton-analysis-design.md"
$designPath = Join-Path $base "docs\design.md"
Write-Host ">> 嘗試從 GitHub 下載最新設計文件..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $rawUrl -OutFile $designPath -UseBasicParsing
    Write-Host "   設計文件已寫入 docs\design.md" -ForegroundColor Green
} catch {
    Write-Host "!! 下載失敗（可能無網路）。已建立佔位檔，請手動複製設計文件。" -ForegroundColor Yellow
    @"
# 設計文件（佔位）

下載失敗，請手動從下列網址複製內容到本檔：
$rawUrl
"@ | Out-File $designPath -Encoding utf8
}

# ---- git init ----
Write-Host ">> 初始化 git..." -ForegroundColor Cyan
Push-Location $base
try {
    if (-not (Test-Path (Join-Path $base ".git"))) {
        git init | Out-Null
        git add -A | Out-Null
        git commit -m "初始化羽球雙打影像分析專案骨架" | Out-Null
        Write-Host "   git 初始化並完成首次 commit" -ForegroundColor Green
    } else {
        Write-Host "   已是 git repo，略過 init" -ForegroundColor DarkGray
    }
} catch {
    Write-Host "!! git 指令失敗（可能未裝 git）；資料夾結構已建好，可稍後再 init。" -ForegroundColor Yellow
}
Pop-Location

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " 完成！專案位置：$base" -ForegroundColor Cyan
Write-Host " 下一步：打開 experiments\exp01_tracknet\README.md 開始階段 0" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
'@
