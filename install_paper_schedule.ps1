$ErrorActionPreference = 'Stop'
$taskName = 'XSMB-Paper-90Day-Trial'
$journalSummary = Join-Path $PSScriptRoot 'reports\paper_trial\summary.json'
if (-not (Test-Path -LiteralPath $journalSummary)) {
    throw 'Run .\run_paper.ps1 init first.'
}
$trial = Get-Content -LiteralPath $journalSummary -Raw -Encoding UTF8 | ConvertFrom-Json
if ((Get-TimeZone).BaseUtcOffset.TotalHours -ne 7 -or (Get-TimeZone).SupportsDaylightSavingTime) {
    throw 'This schedule requires the Windows timezone to be UTC+07:00 without DST.'
}
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output 'Task already exists; left unchanged. Inspect it in Task Scheduler.'
    exit 0
}
$runner = Join-Path $PSScriptRoot 'run_paper.ps1'
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $runner + '" -Command run'
$action = New-ScheduledTaskAction -Execute $powershellPath -Argument $arguments -WorkingDirectory $PSScriptRoot
$startDate = [datetime]::ParseExact($trial.protocol.start, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
$triggers = @(
    (New-ScheduledTaskTrigger -Daily -At ($startDate.AddHours(17))),
    (New-ScheduledTaskTrigger -Daily -At ($startDate.AddHours(19).AddMinutes(30)))
)
foreach ($trigger in $triggers) {
    $trigger.EndBoundary = $trial.protocol.end + 'T23:59:59'
}
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Principal $principal -Settings $settings -Description 'Paper only: frozen Poisson Top 5 trial. Predict 17:00; settle 19:30 UTC+7. No real wagers.' | Select-Object TaskName, State
Write-Output 'Runs only while this Windows user is logged in and the computer is awake. Late runs never backdate forecasts.'
