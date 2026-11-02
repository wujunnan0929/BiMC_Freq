# Stage 2 (one shot at the test set), PowerShell version of final.sh.
#   powershell -File fdrfreq\scripts\final.ps1 -Dataset cub200 -Entries ZS,Proto,TEEN,BiMC,BiMC_ens,A2,K0
param([string]$Dataset, [string[]]$Entries, [string]$Tag = "vitb16", [int[]]$Shots = @(5, 1),
      [int[]]$Seeds = @(1, 2, 3, 4, 5), [string]$Anchor = "A2")
$ErrorActionPreference = "Stop"
$Bank = "banks\${Dataset}_$Tag"; $Out = "outputs\fdrfreq\${Dataset}_$Tag"
foreach ($Shot in $Shots) {
  $D = "$Out\shot$Shot"
  $Locked = $Entries | ForEach-Object { "$D\locked_$_.json" }
  python -m fdrfreq.run evaluate --bank $Bank --shot $Shot --seeds $Seeds --out $D --locked $Locked
  python -m fdrfreq.report --dir $D --anchor $Anchor --shot $Shot | Out-File -Encoding utf8 "$D\tables.md"
  python -m fdrfreq.report --dir $D --anchor $Anchor --shot $Shot --format latex | Out-File -Encoding utf8 "$D\tables.tex"
}
