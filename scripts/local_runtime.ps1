# Shared helpers for the portable Windows runtime. No machine-wide settings.
Set-StrictMode -Version Latest

function Invoke-LocalRuntimeJson {
    param(
        [System.Net.Http.HttpClient]$Client,
        [string]$Uri,
        [int]$TimeoutMs = 1000,
        [object]$Body = $null,
        [string]$ProgressLabel = ''
    )
    $request = $null
    $response = $null
    $cancellation = [System.Threading.CancellationTokenSource]::new()
    $cancellation.CancelAfter([Math]::Max(1, $TimeoutMs))
    try {
        $method = if ($null -eq $Body) { [System.Net.Http.HttpMethod]::Get } else { [System.Net.Http.HttpMethod]::Post }
        $request = [System.Net.Http.HttpRequestMessage]::new($method, $Uri)
        if ($null -ne $Body) {
            $request.Content = [System.Net.Http.StringContent]::new(
                ($Body | ConvertTo-Json -Compress), [System.Text.Encoding]::UTF8, 'application/json'
            )
        }
        $transfer = $Client.SendAsync($request, $cancellation.Token)
        if ($ProgressLabel) {
            $progressTimer = [System.Diagnostics.Stopwatch]::StartNew()
            $nextReport = 10
            while (-not $transfer.IsCompleted) {
                try { $null = $transfer.Wait(500) }
                catch { break }
                if ($progressTimer.Elapsed.TotalSeconds -ge $nextReport) {
                    # Information stream, not the JSON result returned to callers.
                    Write-Host "$($ProgressLabel): $([int]$progressTimer.Elapsed.TotalSeconds)/$([int]($TimeoutMs / 1000)) seconds..."
                    $nextReport += 10
                }
            }
        }
        $response = $transfer.GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) { return $null }
        $json = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return ($json | ConvertFrom-Json)
    }
    catch { return $null }
    finally {
        if ($null -ne $response) { $response.Dispose() }
        if ($null -ne $request) { $request.Dispose() }
        $cancellation.Dispose()
    }
}

function Wait-LocalRuntimeJson {
    param([System.Net.Http.HttpClient]$Client, [string]$Uri, [int]$TimeoutSeconds = 30)
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $remaining = [int][Math]::Max(1, $TimeoutSeconds * 1000 - $timer.Elapsed.TotalMilliseconds)
        $result = Invoke-LocalRuntimeJson -Client $Client -Uri $Uri -TimeoutMs ([Math]::Min(1000, $remaining))
        if ($null -ne $result) { return $result }
        $remaining = [int]($TimeoutSeconds * 1000 - $timer.Elapsed.TotalMilliseconds)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([Math]::Min(200, $remaining)) }
    }
    return $null
}

function Test-LocalRuntimePort {
    param([int]$Port)
    $socket = [System.Net.Sockets.TcpClient]::new()
    try {
        $connected = $socket.ConnectAsync('127.0.0.1', $Port)
        if (-not $connected.Wait(500)) { return $false }
        return $socket.Connected
    }
    catch { return $false }
    finally { $socket.Dispose() }
}

function Get-LocalRuntimeProcess {
    param([int]$ProcessId)
    # Keep CommandLine internal: it must never be written to the console or logs.
    return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
}

function Test-LocalRuntimeProcess {
    param([object]$Process, [string]$Root, [ValidateSet('bot', 'ollama')][string]$Kind, [switch]$IsRoot)
    if ($null -eq $Process -or -not $Process.ExecutablePath -or -not $Process.CommandLine) { return $false }
    try { $executable = [System.IO.Path]::GetFullPath($Process.ExecutablePath) }
    catch { return $false }
    if ($Kind -eq 'bot') {
        $allowed = [System.IO.Path]::Combine($Root, '.venv') + [System.IO.Path]::DirectorySeparatorChar
        return ($executable.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase) -and
            [System.IO.Path]::GetFileName($executable) -match '^pythonw?\.exe$' -and
            $Process.CommandLine -match '(?:^|\s)-m\s+alfalearn(?:\s|$)')
    }
    $allowed = [System.IO.Path]::Combine($Root, '.run', 'ollama') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $executable.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { return $false }
    if ($IsRoot) {
        return ($executable.Equals([System.IO.Path]::Combine($Root, '.run', 'ollama', 'ollama.exe'),
                [StringComparison]::OrdinalIgnoreCase) -and $Process.CommandLine -match '(?:^|\s)serve(?:\s|$)')
    }
    return ($Process.CommandLine -match '(?:^|\s)(?:runner|--model|--ollama-engine)(?:\s|$)')
}

function Get-VerifiedLocalRuntimeStopProcess {
    param(
        [object]$Record, [string]$Root,
        [ValidateSet('bot', 'ollama')][string]$Kind,
        [switch]$IsRoot, [int]$TreeIndex
    )
    $verificationTimer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        $current = Get-LocalRuntimeProcess -ProcessId $Record.ProcessId
        # PID reuse proves that the original target has already exited.
        if ($null -eq $current) { return $null }
        $birthTimeAvailable = $null -ne $current.CreationDate
        if ($birthTimeAvailable -and $current.CreationDate -ne $Record.CreationDate) { return $null }
        if ($birthTimeAvailable -and (Test-LocalRuntimeProcess -Process $current -Root $Root -Kind $Kind -IsRoot:$IsRoot)) {
            return $current
        }
        $metadataAvailable = $birthTimeAvailable -and [bool]$current.ExecutablePath -and [bool]$current.CommandLine
        if ($metadataAvailable -or $verificationTimer.Elapsed.TotalMilliseconds -ge 500) {
            $failure = [System.InvalidOperationException]::new(
                "A $Kind process could not be verified; refusing to stop it."
            )
            $failure.Data['LocalRuntimeStage'] = 'verify-process'
            $failure.Data['LocalRuntimeTarget'] = if ($IsRoot) { 'root' } else { 'child' }
            $failure.Data['LocalRuntimeIndex'] = $TreeIndex
            throw $failure
        }
        # A venv launcher may retain its birth time while its worker exits and
        # CIM temporarily loses path/command metadata. Only re-read; never kill
        # an unverifiable process. Refuse if it remains unverifiable after 500ms.
        $remaining = [int](500 - $verificationTimer.Elapsed.TotalMilliseconds)
        if ($remaining -gt 0) { Start-Sleep -Milliseconds ([Math]::Min(50, $remaining)) }
    }
}

function Stop-LocalRecordedService {
    param([string]$Root, [ValidateSet('bot', 'ollama')][string]$Kind)
    $pidFile = Join-Path $Root ".run\$Kind.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { return }
    $recordedId = 0
    if (-not [int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$recordedId) -or $recordedId -le 0) {
        throw "Invalid $Kind PID record; refusing to stop any process."
    }
    $parent = Get-LocalRuntimeProcess -ProcessId $recordedId
    if ($null -eq $parent) {
        Remove-Item -LiteralPath $pidFile
        Write-Output "$Kind is already stopped."
        return
    }
    if ($null -eq $parent.CreationDate -or -not (Test-LocalRuntimeProcess -Process $parent -Root $Root -Kind $Kind -IsRoot)) {
        throw "The recorded $Kind PID does not identify this project's process; nothing was stopped."
    }
    $snapshot = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $tree = [System.Collections.Generic.List[object]]::new()
    $tree.Add($parent)
    for ($index = 0; $index -lt $tree.Count; $index++) {
        $ancestor = $tree[$index]
        foreach ($child in $snapshot) {
            if ($child.ParentProcessId -eq $ancestor.ProcessId -and $child.CreationDate -ge $ancestor.CreationDate -and
                $child.ProcessId -ne $ancestor.ProcessId -and
                (Test-LocalRuntimeProcess -Process $child -Root $Root -Kind $Kind)) {
                $tree.Add($child)
            }
        }
    }
    for ($index = $tree.Count - 1; $index -ge 0; $index--) {
        $record = $tree[$index]
        $current = Get-VerifiedLocalRuntimeStopProcess -Record $record -Root $Root -Kind $Kind `
            -IsRoot:($index -eq 0) -TreeIndex $index
        if ($null -eq $current) { continue }
        try { Stop-Process -Id $current.ProcessId -Force -ErrorAction Stop }
        catch {
            # Killing a Python worker can also finish its venv launcher between
            # the CIM identity check and Stop-Process. That is a successful stop.
            $afterFailure = Get-LocalRuntimeProcess -ProcessId $record.ProcessId
            if ($null -eq $afterFailure -or
                ($null -ne $afterFailure.CreationDate -and $afterFailure.CreationDate -ne $record.CreationDate)) { continue }
            throw "A verified $Kind process could not be stopped."
        }
    }
    if ((Get-Content -LiteralPath $pidFile -Raw).Trim() -eq [string]$recordedId) {
        Remove-Item -LiteralPath $pidFile
    }
    Write-Output "Stopped this project's $Kind process tree."
}
