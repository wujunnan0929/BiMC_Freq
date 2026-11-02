# Stage 1 (development only), PowerShell version of dev.sh.
#   powershell -ExecutionPolicy Bypass -File fdrfreq\scripts\dev.ps1 -Dataset cub200 -DataRoot ..\datasets -Gpu 0
param([string]$Dataset, [string]$DataRoot, [string]$Gpu = "0", [string]$Backbone = "ViT-B/16",
      [string]$Tag = "vitb16", [int[]]$Shots = @(5, 1), [int]$Sequences = 10)
$ErrorActionPreference = "Stop"
$Dct = if ($Dataset -eq "cub200") { @(6, 9, 12) } else { @(9) }
$Bank = "banks\${Dataset}_$Tag"; $Out = "outputs\fdrfreq\${Dataset}_$Tag"
if (-not (Test-Path "$Bank\meta.json")) {
  python -m fdrfreq.extract --dataset $Dataset --data-root $DataRoot --out $Bank --backbone $Backbone `
    --gpu $Gpu --dct-layers $Dct --sigmas 1 2 4 --aug flip gray --seeds 1 2 3 4 5 --keywords
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
python -m fdrfreq.run parity --bank $Bank --seeds 1 2 3 4 5 --out $Out
foreach ($Shot in $Shots) {
  $D = "$Out\shot$Shot"
  python -m fdrfreq.run diagnose --bank $Bank --shot $Shot --out $D
  python -m fdrfreq.run select --bank $Bank --shot $Shot --sequences $Sequences --matrix fdrfreq\configs\paper.json --out $D
  python -m fdrfreq.run gates --dir $D
}
