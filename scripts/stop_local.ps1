# Stop only verified processes recorded for this workspace, never arbitrary Python.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'local_runtime.ps1')
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskFailures = [System.Collections.Generic.List[string]]::new()
foreach ($kind in @('bot', 'ollama')) {
    try { Stop-LocalRecordedService -Root $taskRoot -Kind $kind }
    catch {
        # Do not expose exception details containing process command lines.
        $taskErrorType = $_.Exception.GetType().Name
        $taskErrorLine = $_.InvocationInfo.ScriptLineNumber
        $taskErrorStage = 'inspection-or-stop'
        $taskErrorTarget = 'unknown'
        $taskErrorIndex = 'unknown'
        if ($_.Exception.Data.Contains('LocalRuntimeStage')) {
            $taskErrorStage = $_.Exception.Data['LocalRuntimeStage']
            $taskErrorTarget = $_.Exception.Data['LocalRuntimeTarget']
            $taskErrorIndex = $_.Exception.Data['LocalRuntimeIndex']
        }
        Write-Warning "Could not safely stop $kind (stage: $taskErrorStage, target: $taskErrorTarget, index: $taskErrorIndex, error type: $taskErrorType, source line: $taskErrorLine). Its PID record may be stale or process inspection may be unavailable."
        $taskFailures.Add($kind)
    }
}
if ($taskFailures.Count -gt 0) { throw 'Some processes were left running because ownership could not be verified.' }
