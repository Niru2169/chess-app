<#
  laptop_train.ps1 — train the 3 distill modes on the local GPU (RTX 4060).

  Fallback for when no cloud VM is available. Uses the same defaults as the
  original training command: sample-rate 0.1, epochs 3, batch-size 256.
  Runs the 3 modes sequentially on the single local GPU.

  Run:
    powershell -ExecutionPolicy Bypass -File laptop_train.ps1
  Override:
    powershell -ExecutionPolicy Bypass -File laptop_train.ps1 -Epochs 4 -SampleRate 0.2
  Everything is a CLI flag on preprocess.py / train.py; the defaults below match
  what the original `preprocess.py games.pgn --sample-rate 0.1 --workers 20`
  and `train.py --data data --epochs 3 --batch-size 256 --save chessgnn.pt` used.
#>
[CmdletBinding()]
param(
  [double]$SampleRate   = 0.1,
  [int]$Epochs          = 3,
  [int]$BatchSize       = 256,
  [int]$DistillEvery    = 300,
  [int]$DistillBoards   = 8,
  [int]$Simulations     = 32,
  [int]$Subgraphs       = 2,
  [switch]$SkipPreprocess
)

$ErrorActionPreference = 'Stop'
$repo   = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Join-Path $repo 'BayesianELO-main'
$py     = Join-Path $repo 'autoChess\.venv\Scripts\python.exe'
$shards = Join-Path $engine 'data'
$runs   = Join-Path $engine 'runs'
$pgnDir = Join-Path $repo 'training_data'

if (-not (Test-Path -LiteralPath $py)) { throw "venv python not found: $py" }

function Say($msg) { Write-Output $msg }

Say "=== laptop training: sample-rate $SampleRate, epochs $Epochs, batch $BatchSize ==="
Say "engine: $engine"
$torchInfo = & $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
Say "torch: $torchInfo"

# locate (or extract) the dataset PGN; the filename contains [1967497]
$pgn = Get-ChildItem -LiteralPath $pgnDir -Filter 'CCRL-4040*.pgn' -ErrorAction SilentlyContinue |
       Select-Object -First 1
if (-not $pgn) {
  $archive = Get-ChildItem -LiteralPath $repo -Filter 'CCRL-4040*.7z' -ErrorAction SilentlyContinue |
             Select-Object -First 1
  if ($archive) {
    Say "extracting $($archive.Name)"
    New-Item -ItemType Directory -Force -Path $pgnDir | Out-Null
    & 'C:\Users\User\scoop\shims\7z.exe' x $archive.FullName "-o$pgnDir" -y | Out-Null
    $pgn = Get-ChildItem -LiteralPath $pgnDir -Filter 'CCRL-4040*.pgn' | Select-Object -First 1
  }
}
if (-not $pgn) { throw "dataset not found; extract the CCRL 7z into $pgnDir" }
Say "dataset: $($pgn.FullName)"

New-Item -ItemType Directory -Force -Path $shards, $runs | Out-Null

Say "=== [1/2] preprocess (sample-rate $SampleRate) ==="
if (-not $SkipPreprocess) {
  & $py (Join-Path $engine 'gnn-chess\preprocess.py') $pgn.FullName --out $shards `
       --sample-rate $SampleRate --workers ([Math]::Max(1, $env:NUMBER_OF_PROCESSORS - 4))
  if ($LASTEXITCODE -ne 0) { throw "preprocess failed" }
}
Say "shards: $((Get-ChildItem -LiteralPath $shards -Filter 'shard_*.npz').Count)"

Say "=== [2/2] train 3 modes sequentially (epochs $Epochs) ==="
Push-Location $engine
foreach ($mode in @('minimax', 'mcts', 'both')) {
  Say "--- $mode ---"
  & $py gnn-chess\train.py --data $shards --epochs $Epochs --batch-size $BatchSize `
       --device cuda --save (Join-Path $runs "model_$mode.pt") `
       --distill-mode $mode --distill-every $DistillEvery --distill-boards $DistillBoards `
       --simulations $Simulations --subgraphs $Subgraphs --ignore_duplicates
  if ($LASTEXITCODE -ne 0) { Say "  $mode FAILED (exit $LASTEXITCODE)" }
}
Pop-Location

Say "=== done. checkpoints: ==="
Get-ChildItem -LiteralPath $runs -Filter '*.pt' -ErrorAction SilentlyContinue |
  Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,2)}}
