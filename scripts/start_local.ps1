# Start the already-installed portable Ollama and this project's Telegram bot.
param([ValidateRange(1, 600)][int]$WarmupTimeoutSeconds = 120)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'local_runtime.ps1')
Add-Type -AssemblyName System.Net.Http

$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskOllama = Join-Path $taskRoot '.run\ollama\ollama.exe'
$taskRuntime = Join-Path $taskRoot '.run'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Install the project Python environment first.' }
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot '.env') -PathType Leaf)) { throw 'Configure the local .env file first.' }
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null

Push-Location $taskRoot
try {
    # Only nonsensitive settings are serialized. Neither secret is printed.
    $taskSettingsJson = & $taskPython -c "import json; from alfalearn.config import Settings; s=Settings(); print(json.dumps({'host':s.host,'port':s.port,'ollama_enabled':s.ollama_enabled,'ollama_base_url':str(s.ollama_base_url),'ollama_model':s.ollama_model,'has_ollama_key':bool(s.ollama_api_key and s.ollama_api_key.get_secret_value())}))"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read project settings.' }
    $taskSettings = $taskSettingsJson | ConvertFrom-Json
}
finally { Pop-Location }
if (-not $taskSettings.ollama_enabled -or $taskSettings.ollama_base_url.TrimEnd('/') -ne 'http://127.0.0.1:11434' -or $taskSettings.has_ollama_key) {
    throw 'Configure .env for local Ollama: enabled=true, base_url=http://127.0.0.1:11434 and an empty OLLAMA_API_KEY.'
}
if ($taskSettings.host -notin @('127.0.0.1', '0.0.0.0', 'localhost')) { throw 'The local launcher requires a loopback-compatible HTTP HOST.' }

$taskEnvironment = @{
    OLLAMA_MODELS = (Join-Path $taskRoot '.run\ollama-models')
    OLLAMA_HOST = '127.0.0.1:11434'
    OLLAMA_NO_CLOUD = '1'
    OLLAMA_CONTEXT_LENGTH = '16384'
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_KEEP_ALIVE = '-1'
    PYTHONIOENCODING = 'utf-8'
}
$taskPreviousEnvironment = @{}
$taskHandler = [System.Net.Http.HttpClientHandler]::new()
$taskHandler.UseProxy = $false
$taskHttp = [System.Net.Http.HttpClient]::new($taskHandler)
$taskHttp.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan
try {
    foreach ($name in $taskEnvironment.Keys) {
        $taskPreviousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $taskEnvironment[$name], 'Process')
    }
    $taskTags = Invoke-LocalRuntimeJson -Client $taskHttp -Uri 'http://127.0.0.1:11434/api/tags'
    if ($null -eq $taskTags) {
        if (Test-LocalRuntimePort -Port 11434) { throw 'Port 11434 is occupied but did not return an Ollama model list.' }
        if (-not (Test-Path -LiteralPath $taskOllama -PathType Leaf)) { throw 'Portable Ollama is missing at .run/ollama/ollama.exe.' }
        $taskOllamaProcess = Start-Process -FilePath $taskOllama -ArgumentList 'serve' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $taskRuntime 'ollama.stdout.log') -RedirectStandardError (Join-Path $taskRuntime 'ollama.stderr.log')
        $taskOllamaProcess.Id | Set-Content -LiteralPath (Join-Path $taskRuntime 'ollama.pid')
        $taskTags = Wait-LocalRuntimeJson -Client $taskHttp -Uri 'http://127.0.0.1:11434/api/tags'
        if ($null -eq $taskTags) { throw 'Ollama did not become ready within 30 seconds. See .run/ollama.stderr.log.' }
        Write-Output 'Started portable Ollama.'
    }
    else { Write-Output 'Reusing Ollama already listening on 127.0.0.1:11434.' }
    if ($null -eq $taskTags.PSObject.Properties['models'] -or
        @($taskTags.models | Where-Object { $_.name -eq $taskSettings.ollama_model }).Count -eq 0) {
        throw 'The configured model is not downloaded. This launcher does not install or download models.'
    }
    Write-Output "Loading the configured model into memory (up to $WarmupTimeoutSeconds seconds; the first CUDA warmup may take longer)..."
    $taskWarmup = Invoke-LocalRuntimeJson -Client $taskHttp -Uri 'http://127.0.0.1:11434/api/generate' -TimeoutMs ($WarmupTimeoutSeconds * 1000) `
        -Body @{model = $taskSettings.ollama_model; keep_alive = -1; stream = $false} -ProgressLabel 'Model warmup'
    if ($null -eq $taskWarmup -or $null -eq $taskWarmup.PSObject.Properties['done'] -or -not $taskWarmup.done) {
        throw "Model warmup failed or exceeded $WarmupTimeoutSeconds seconds. See .run/ollama.stderr.log."
    }
    $taskHealthUri = "http://127.0.0.1:$($taskSettings.port)/health"
    $taskHealth = Invoke-LocalRuntimeJson -Client $taskHttp -Uri $taskHealthUri
    if ($null -eq $taskHealth) {
        if (Test-LocalRuntimePort -Port $taskSettings.port) { throw 'The project HTTP port is occupied by an unrecognized service.' }
        $taskBotProcess = Start-Process -FilePath $taskPython -ArgumentList '-u', '-m', 'alfalearn' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $taskRuntime 'bot.stdout.log') -RedirectStandardError (Join-Path $taskRuntime 'bot.stderr.log')
        $taskBotProcess.Id | Set-Content -LiteralPath (Join-Path $taskRuntime 'bot.pid')
        $taskHealth = Wait-LocalRuntimeJson -Client $taskHttp -Uri $taskHealthUri
        if ($null -eq $taskHealth) { throw 'The bot did not become ready within 30 seconds. See .run/bot.stderr.log.' }
        Write-Output 'Started the project bot in the background.'
    }
    else { Write-Output 'The project HTTP server is already running; no second bot was started.' }
    if ($taskHealth.status -ne 'ok' -or $null -eq $taskHealth.PSObject.Properties['llm'] -or
        $taskHealth.llm.provider -ne 'ollama' -or $taskHealth.llm.model -ne $taskSettings.ollama_model) {
        throw 'The running HTTP server does not report the expected local model configuration. Stop this project before restarting.'
    }
    Write-Output "Ready: http://127.0.0.1:$($taskSettings.port)/docs"
    Write-Output 'Stop with: .\scripts\stop_local.ps1'
}
finally {
    foreach ($name in $taskPreviousEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $taskPreviousEnvironment[$name], 'Process')
    }
    $taskHttp.Dispose()
}
