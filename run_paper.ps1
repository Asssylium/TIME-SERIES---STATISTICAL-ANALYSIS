param(
    [ValidateSet('init', 'run', 'status', 'verify')]
    [string]$Command = 'run'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = 'C:\Program Files\PostgreSQL\18\pgAdmin 4\python\python.exe'
$entry = @'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, 'D:/xsmb-model-check/deps')
sys.path.insert(0, os.getcwd())
from scripts.operations.paper_trade import main
raise SystemExit(main())
'@
$logDirectory = Join-Path $PSScriptRoot 'reports\paper_trial\logs'
[void](New-Item -ItemType Directory -Path $logDirectory -Force)
$logPath = Join-Path $logDirectory ((Get-Date -Format 'yyyy-MM-dd_HHmmss_fff') + '.log')
# All failures are logged and propagated to Task Scheduler.
& $pythonPath -c $entry $Command 2>&1 | Tee-Object -FilePath $logPath
exit $LASTEXITCODE
