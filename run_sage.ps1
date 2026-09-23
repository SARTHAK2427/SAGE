# Starts SAGE using the local Python runtime where the project dependencies
# are installed.  Invoke from PowerShell with: .\run_sage.ps1
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = 'C:\Users\sunbu\AppData\Local\Programs\Python\Python313\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Required Python runtime was not found: $pythonExe"
}

Set-Location -LiteralPath $projectRoot
& $pythonExe app.py
exit $LASTEXITCODE
