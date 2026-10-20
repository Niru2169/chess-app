<#
  watch_train.ps1 - live training dashboard (READ-ONLY).

  Tails a training log, parses the lines train.py / preprocess.py already print,
  and redraws one screen with a progress bar, ETA, losses, and GPU usage.

  It never writes to the log, never signals the training process, and never
  imports the training code. Safe to start/stop at any time.

  Run:
    powershell -ExecutionPolicy Bypass -File watch_train.ps1
    powershell -ExecutionPolicy Bypass -File watch_train.ps1 -Log C:\path\to\run.log
    powershell -ExecutionPolicy Bypass -File watch_train.ps1 -Replay      # from the beginning

  The training has to be writing to -Log for the bar to move. If you launched it
  in a plain console window, nothing is on disk; start the next run like this:
    powershell -ExecutionPolicy Bypass -File laptop_train.ps1 2>&1 |
      Tee-Object -FilePath laptop_train.log
  and then run this script in a second window.

  Ctrl+C stops watching. It does not stop training.
#>
[CmdletBinding()]
param(
  [string]$Log = (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'laptop_train.log'),
  [int]$Interval = 1,
  [switch]$Replay,
  [switch]$NoGpu
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- state

$script:st = @{
  Stage       = 'waiting'
  Mode        = ''
  Step        = 0
  Total       = 0
  Loss        = $null
  NextMove    = $null
  Lr          = ''
  ValLoss     = $null
  DistillLoss = $null
  TrainPos    = $null
  ValPos      = $null
  PreChunks   = 0
  PrePos      = $null
  StepStart   = $null
  Started     = Get-Date
  Events      = New-Object System.Collections.Generic.List[string]
}

$script:offset       = 0
$script:partial      = ''
$script:lastFrame    = 0
$script:procInfo     = ''
$script:gpuInfo      = ''
$script:tick         = 0
$script:lastStep     = 0
$script:lastStepTime = $null
$script:rate         = 0.0
$script:watchStart   = Get-Date

# the machine locale may group digits non-western (e.g. en-IN); keep output invariant
$script:inv = [System.Globalization.CultureInfo]::InvariantCulture

function Fmt([string]$fmt) {
  $v = @($args)
  return [string]::Format($script:inv, $fmt, $v)
}

function Add-Event([string]$text) {
  if (-not $text) { return }
  $script:st.Events.Add($text)
  while ($script:st.Events.Count -gt 7) { $script:st.Events.RemoveAt(0) }
}

function Fmt-Span($ts) {
  if (-not $ts) { return '--:--' }
  $s = [int]((Get-Date) - $ts).TotalSeconds
  if ($s -lt 0) { $s = 0 }
  $h = [int]($s / 3600); $m = [int](($s % 3600) / 60); $sec = $s % 60
  if ($h -gt 0) { return ('{0}:{1:00}:{2:00}' -f $h, $m, $sec) }
  return ('{0:00}:{1:00}' -f $m, $sec)
}

function Fmt-Num($n) {
  if ($null -eq $n) { return '--' }
  return (Fmt '{0:N0}' ([double]$n))
}

# ---------------------------------------------------------------- parsing

function Parse-Line([string]$line) {
  $st = $script:st

  if ($line -match '^step (\d+)/(\d+)\s+loss ([\d.]+)\s+next_move ([\d.]+)\s+lr (\S+)') {
    if (-not $st.StepStart) { $st.StepStart = Get-Date }
    $newStep = [int]$matches[1]
    $now = Get-Date
    if ($script:lastStepTime -and $newStep -gt $script:lastStep) {
      $dt = ($now - $script:lastStepTime).TotalSeconds
      if ($dt -ge 1.0) {
        $inst = ($newStep - $script:lastStep) / $dt
        if ($script:rate -le 0) { $script:rate = $inst }
        else { $script:rate = (0.6 * $script:rate) + (0.4 * $inst) }
      }
    }
    $script:lastStep = $newStep
    $script:lastStepTime = $now
    $st.Step     = $newStep
    $st.Total    = [int]$matches[2]
    $st.Loss     = [double]$matches[3]
    $st.NextMove = [double]$matches[4]
    $st.Lr       = $matches[5]
    return
  }
  if ($line -match '^\s*\[val\] loss ([\d.]+)') {
    $st.ValLoss = [double]$matches[1]
    Add-Event ('val loss ' + $matches[1])
    return
  }
  if ($line -match '^\s*\[distill:([\w]+)\] loss ([\d.]+)') {
    $st.DistillLoss = [double]$matches[2]
    return
  }
  if ($line -match '^--- (\w+) ---\s*$') {
    $st.Mode = $matches[1]
    $st.Step = 0; $st.Total = 0; $st.StepStart = $null
    $st.Loss = $null; $st.NextMove = $null; $st.ValLoss = $null; $st.DistillLoss = $null
    $st.Stage = 'train'
    $script:lastStep = 0; $script:lastStepTime = $null; $script:rate = 0.0
    Add-Event ('started mode: ' + $matches[1])
    return
  }
  if ($line -match '^=== (.+?) ===\s*$') {
    $st.Stage = $matches[1]
    Add-Event $matches[1]
    return
  }
  if ($line -match 'chunks=(\d+) positions=([\d,]+)') {
    $st.PreChunks = [int]$matches[1]
    $st.PrePos = $matches[2].Replace(',', '')
    $st.Stage = 'preprocess'
    return
  }
  if ($line -match 'train positions: ([\d,]+)\s+val positions: ([\d,]+)') {
    $st.TrainPos = $matches[1].Replace(',', '')
    $st.ValPos = $matches[2].Replace(',', '')
    Add-Event ('loaded ' + $matches[1] + ' train / ' + $matches[2] + ' val positions')
    return
  }
  if ($line -match '^saved (.+?)\s*$') {
    Add-Event ('saved ' + (Split-Path -Leaf $matches[1]))
    return
  }
  if ($line -match '^(loading data|not enough data)') {
    Add-Event $line.Trim()
    return
  }
  if ($line -match '(Traceback|CUDA out of memory|RuntimeError|FAILED|Error:)') {
    Add-Event $line.Trim()
    return
  }
  if ($line -match '^\s+(\S.*)$' -and $line.Trim().Length -gt 0 -and $line -notmatch '^\[?\s*\d+s\]') {
    # indented chatter: keep a little of it
    Add-Event $line.Trim()
  }
}

function Read-NewLines {
  if (-not (Test-Path -LiteralPath $Log)) { return @() }
  $len = (Get-Item -LiteralPath $Log).Length
  if ($len -lt $script:offset) { $script:offset = 0; $script:partial = '' }
  if ($len -eq $script:offset) { return @() }

  $n = 0
  $buf = $null
  $fs = [System.IO.File]::Open($Log, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
  try {
    [void]$fs.Seek($script:offset, [System.IO.SeekOrigin]::Begin)
    $want = [int]($len - $script:offset)
    $buf = New-Object byte[] $want
    $n = $fs.Read($buf, 0, $want)
    $script:offset += $n
  } finally { $fs.Dispose() }

  if ($n -le 0) { return @() }
  $text = $script:partial + [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
  $parts = $text -split "`r?`n"
  $script:partial = $parts[$parts.Count - 1]
  if ($parts.Count -le 1) { return @() }
  return $parts[0..($parts.Count - 2)] | Where-Object { $_.Trim().Length -gt 0 }
}

# ---------------------------------------------------------------- probes

function Update-Probe {
  $script:tick++
  if ($script:tick % 3 -ne 1) { return }

  $script:procInfo = 'no train.py process'
  try {
    $cands = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
      Where-Object { $_.CommandLine -and $_.CommandLine -match 'train\.py' })
    $best = $null; $bestCpu = -1
    foreach ($c in $cands) {
      $proc = Get-Process -Id $c.ProcessId -ErrorAction SilentlyContinue
      if (-not $proc) { continue }
      if ($proc.CPU -gt $bestCpu) { $best = $proc; $bestCpu = $proc.CPU }
    }
    if ($best) {
      $script:procInfo = (Fmt 'pid {0}   up {1}   cpu {2:N0}s' $best.Id (Fmt-Span $best.StartTime) $best.CPU)
    }
  } catch { $script:procInfo = 'process probe failed' }

  $script:gpuInfo = 'gpu: n/a'
  if (-not $NoGpu) {
    try {
      $g = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
            --format=csv,noheader,nounits 2>$null | Select-Object -First 1
      if ($g) {
        $a = $g -split ',\s*'
        $script:gpuInfo = ('gpu {0}%  vram {1}/{2} MB  {3}C' -f $a[0], $a[1], $a[2], $a[3])
      }
    } catch { $script:gpuInfo = 'gpu: n/a' }
  }
}

# ---------------------------------------------------------------- render

function Bar([double]$frac, [int]$width) {
  if ($frac -lt 0) { $frac = 0 }
  if ($frac -gt 1) { $frac = 1 }
  $fill = [int][math]::Round($frac * $width)
  if ($fill -gt $width) { $fill = $width }
  return ('[' + ('#' * $fill) + ('.' * ($width - $fill)) + ']')
}

function Render {
  $st = $script:st
  $w = 96
  $h = 0
  try {
    if ([Console]::WindowWidth -gt 40) { $w = [Console]::WindowWidth - 1 }
    $h = [Console]::WindowHeight
  } catch { }
  if ($h -lt 20) { $h = 40 }
  $room = [math]::Max(0, [math]::Min(7, $h - 18))

  $out = New-Object System.Collections.Generic.List[string]
  $out.Add('')
  $out.Add('  CHESS GNN TRAINING - live view (read-only)')
  $out.Add('  ' + ('-' * ($w - 4)))
  $out.Add('  log      : ' + $Log)
  $exists = Test-Path -LiteralPath $Log
  $out.Add('  log size : ' + $(if ($exists) { Fmt-Num (Get-Item -LiteralPath $Log).Length } else { 'missing' }) +
            ' bytes   updated ' + $(if ($exists) { (Get-Item -LiteralPath $Log).LastWriteTime.ToString('HH:mm:ss') } else { '--' }))
  $out.Add('  stage    : ' + $st.Stage + $(if ($st.Mode) { '   mode: ' + $st.Mode } else { '' }))
  $out.Add('  ' + $script:procInfo)
  $out.Add('  ' + $script:gpuInfo)
  $out.Add('')

  if ($st.Total -gt 0) {
    $frac = $st.Step / $st.Total
    $eta = '--:--'
    if ($script:rate -gt 0 -and $frac -lt 1) {
      $eta = Fmt-Span ((Get-Date).AddSeconds(($st.Total - $st.Step) / $script:rate))
    }
    $out.Add((Fmt '  {0} {1,6:N1}%   step {2} / {3}' (Bar $frac 44) ($frac * 100) $st.Step $st.Total))
    $out.Add((Fmt '  watching {0}   eta {1}   {2:N2} steps/s' (Fmt-Span $script:watchStart) $eta $script:rate))
    $out.Add('')
    $out.Add((Fmt '  loss {0}   next_move {1}   lr {2}' `
      $(if ($null -ne $st.Loss) { Fmt '{0:N4}' $st.Loss } else { '--' }) `
      $(if ($null -ne $st.NextMove) { Fmt '{0:N3}' $st.NextMove } else { '--' }) `
      $(if ($st.Lr) { $st.Lr } else { '--' })))
    $out.Add((Fmt '  val loss {0}   distill loss {1}' `
      $(if ($null -ne $st.ValLoss) { Fmt '{0:N4}' $st.ValLoss } else { '--' }) `
      $(if ($null -ne $st.DistillLoss) { Fmt '{0:N4}' $st.DistillLoss } else { '--' })))
    if ($st.TrainPos) {
      $out.Add('  positions: ' + (Fmt-Num $st.TrainPos) + ' train / ' + (Fmt-Num $st.ValPos) + ' val')
    }
  }
  elseif ($st.Stage -eq 'preprocess') {
    $out.Add('  preprocessing (no known total)')
    $out.Add('  chunks ' + $st.PreChunks + '   positions ' + (Fmt-Num $st.PrePos) + '   elapsed ' + (Fmt-Span $st.Started))
  }
  else {
    $out.Add('  waiting for progress lines...')
    if (-not $exists) {
      $out.Add('')
      $out.Add('  This log file does not exist yet. Training launched in a plain')
      $out.Add('  console window writes to the screen only, not to disk.')
      $out.Add('  For the next run:')
      $out.Add('    powershell -ExecutionPolicy Bypass -File laptop_train.ps1 2>&1 |')
      $out.Add('      Tee-Object -FilePath laptop_train.log')
    }
  }

  $out.Add('')
  $out.Add('  recent:')
  if ($st.Events.Count -eq 0) { $out.Add('    (nothing yet)') }
  else {
    $keep = @($st.Events)
    if ($keep.Count -gt $room) { $keep = $keep[($keep.Count - $room)..($keep.Count - 1)] }
    foreach ($e in $keep) {
      $s = $e
      if ($s.Length -gt ($w - 8)) { $s = $s.Substring(0, $w - 8) }
      $out.Add('    ' + $s)
    }
  }
  $out.Add('')
  $out.Add('  Ctrl+C to close this viewer (training keeps running)')

  $padded = $out | ForEach-Object {
    if ($_.Length -ge $w) { $_.Substring(0, $w) } else { $_.PadRight($w) }
  }
  while ($padded.Count -lt $script:lastFrame) { $padded += (' ' * $w) }
  $script:lastFrame = $out.Count

  try {
    try {
      [Console]::SetCursorPosition(0, 0)
    } catch {
      Clear-Host
    }
    foreach ($l in $padded) { [Console]::Write($l + "`n") }
  } catch {
    foreach ($l in $padded) { Write-Output $l }
  }
}

# ---------------------------------------------------------------- main

if (-not $Replay -and (Test-Path -LiteralPath $Log)) {
  # start near the end of an existing log instead of replaying all of it
  $len = (Get-Item -LiteralPath $Log).Length
  $script:offset = [math]::Max(0, $len - 20000)
}

try { [Console]::CursorVisible = $false } catch { }

try {
  while ($true) {
    $new = Read-NewLines
    foreach ($line in $new) { Parse-Line $line }
    Update-Probe
    try { Render } catch { [Console]::Error.WriteLine($_.Exception.ToString()) }
    Start-Sleep -Seconds ([math]::Max(1, $Interval))
  }
}
finally {
  try { [Console]::CursorVisible = $true } catch { }
  Write-Output ''
  Write-Output 'viewer closed - training was not touched'
}
