# Inspect which C++ runtime DLLs a running process actually loaded, and from where.
# No ProcMon, no admin rights. ASCII-only on purpose: Windows PowerShell 5.1 reads
# .ps1 files as ANSI unless they carry a UTF-8 BOM, so non-ASCII text here would
# be mangled and break the script.
#
# Usage:
#   powershell -NoProfile -File tools\inspect_loaded_modules.ps1
#   powershell -NoProfile -File tools\inspect_loaded_modules.ps1 -ProcessName 'BlindWatermarkGUI_v0.2.11_build.d161aaa'
param(
    [string]$ProcessName = 'BlindWatermarkGUI*'
)

$ErrorActionPreference = 'Stop'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }

$procs = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if (-not $procs) {
    Write-Output "[not found] no process matches '$ProcessName'. Start the app first, then run this script."
    exit 1
}

$watch = @(
    'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll',
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
    'concrt140.dll', 'vccorlib140.dll', 'vcamp140.dll', 'vcomp140.dll',
    'ucrtbase.dll'
)

foreach ($proc in $procs) {
    Write-Output ''
    Write-Output ('=== PID {0}  {1} ===' -f $proc.Id, $proc.ProcessName)
    try {
        $mods = $proc.Modules
    } catch {
        Write-Output ('  [skip] cannot read module list: {0}' -f $_.Exception.Message)
        continue
    }

    $exeDir = ''
    try { $exeDir = Split-Path $proc.Path -Parent } catch { }

    $rows = foreach ($m in $mods) {
        if ($watch -contains $m.ModuleName.ToLower()) {
            [pscustomobject]@{
                Module = $m.ModuleName
                Path = $m.FileName
                Origin = $(if (($m.FileName -match '\\_MEI\d+') -or ($exeDir -and $m.FileName.StartsWith($exeDir, 'OrdinalIgnoreCase'))) { 'BUNDLED' } else { 'SYSTEM' })
            }
        }
    }

    if (-not $rows) {
        Write-Output '  [none] no watched runtime DLLs found in this process.'
        continue
    }
    foreach ($row in $rows | Sort-Object Module) {
        Write-Output ('  {0,-28} [{1,-7}] {2}' -f $row.Module, $row.Origin, $row.Path)
    }
}

Write-Output ''
Write-Output 'Expected: C++ runtime (vcruntime140/msvcp140/concrt140/...) shows [BUNDLED].'
Write-Output 'ucrtbase.dll showing [SYSTEM] is normal: the UCRT ships with Windows 10+.'
