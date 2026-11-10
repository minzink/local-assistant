# Flay unter Windows einrichten: Programme (uv, Git LFS, Ollama), Python-Pakete, Modelle.
# Kann beliebig oft ausgefuehrt werden; was schon da ist, wird uebersprungen.
#   scripts\setup.cmd                  (Doppelklick oder aus cmd/PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-NoInstall] [-SkipModels]
# Jeder Lauf schreibt ein ausfuehrliches Log nach logs\setup-<Zeitstempel>.log.
# ASCII only: Windows PowerShell 5.1 reads scripts without BOM as ANSI.
param(
    [switch]$NoInstall,   # fehlende Programme nicht per winget installieren, nur pruefen
    [switch]$SkipModels   # 'flay setup' (Modelle laden) auslassen
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("setup-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))
$OllamaUrl = "http://127.0.0.1:11434"

# Python prints UTF-8 (checkmarks, umlauts); piped into PowerShell it would otherwise use
# the ANSI code page and crash with UnicodeEncodeError.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:NO_COLOR = "1"
$env:UV_NO_PROGRESS = "1"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $line = "{0:HH:mm:ss} {1,-5} {2}" -f (Get-Date), $Level, $Message
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    switch ($Level) {
        "DEBUG" { }  # only in the log file
        "STEP"  { Write-Host "`n==> $Message" -ForegroundColor Cyan }
        "OK"    { Write-Host "  OK  $Message" -ForegroundColor Green }
        "WARN"  { Write-Host "  !!  $Message" -ForegroundColor Yellow }
        "ERROR" { Write-Host "  XX  $Message" -ForegroundColor Red }
        default { Write-Host "      $Message" }
    }
}

# Runs a program, shows and logs its output (stdout and stderr), returns the exit code.
function Invoke-Logged {
    param([string]$File, [string[]]$Arguments = @(), [switch]$Quiet)
    Write-Log ("> {0} {1}" -f $File, ($Arguments -join " ")) "DEBUG"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"  # stderr lines must not abort the script
    $global:LASTEXITCODE = 0
    try {
        & $File @Arguments 2>&1 | ForEach-Object {
            $text = "$_"
            Add-Content -Path $LogFile -Value "    $text" -Encoding UTF8
            if (-not $Quiet) { Write-Host "    $text" }
        }
        $code = $global:LASTEXITCODE
    } catch {
        Write-Log "$File konnte nicht gestartet werden: $($_.Exception.Message)" "ERROR"
        $code = -1
    } finally {
        $ErrorActionPreference = $previous
    }
    Write-Log "  Exit-Code $code" "DEBUG"
    return $code
}

# Programs installed by winget land in PATH only for new shells.
function Update-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $extra = "$env:USERPROFILE\.local\bin;$env:LOCALAPPDATA\Programs\Ollama;$env:ProgramFiles\Git\cmd"
    $env:Path = "$machine;$user;$extra"
}

function Find-Program([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    return $null
}

# Git LFS comes with Git for Windows, but not as git-lfs.exe in PATH: ask git itself.
function Find-GitLfs {
    if (-not (Find-Program "git")) { return $null }
    $version = (& git lfs version 2>$null) -join " "
    if ($LASTEXITCODE -eq 0 -and $version) { return $version }
    return $null
}

function Install-Program {
    param([string]$Command, [string]$WingetId, [string]$Name)
    $found = if ($Command -eq "git-lfs") { Find-GitLfs } else { Find-Program $Command }
    if ($found) {
        Write-Log "$Name gefunden: $found" "OK"
        return $true
    }
    if ($NoInstall) {
        Write-Log "$Name fehlt (winget install --id $WingetId)" "ERROR"
        return $false
    }
    if (-not (Find-Program "winget")) {
        Write-Log "$Name fehlt und winget ist nicht verfuegbar - bitte $Name von Hand installieren" "ERROR"
        return $false
    }
    Write-Log "$Name wird installiert (winget: $WingetId) ..."
    $code = Invoke-Logged "winget" @("install", "--id", $WingetId, "--exact", "--silent",
        "--accept-source-agreements", "--accept-package-agreements", "--disable-interactivity")
    Update-Path
    $found = if ($Command -eq "git-lfs") { Find-GitLfs } else { Find-Program $Command }
    if ($found) {
        Write-Log "$Name installiert: $found" "OK"
        return $true
    }
    Write-Log "$Name konnte nicht installiert werden (winget Exit-Code $code)" "ERROR"
    return $false
}

# HTTP GET without any proxy (like Flay since the proxy fix). Returns the body or $null.
function Get-Direct([string]$Url, [int]$TimeoutMs = 3000) {
    try {
        $request = [System.Net.WebRequest]::Create($Url)
        $request.Proxy = $null
        $request.Timeout = $TimeoutMs
        $response = $request.GetResponse()
        try {
            $reader = New-Object System.IO.StreamReader($response.GetResponseStream())
            return $reader.ReadToEnd()
        } finally {
            $response.Close()
        }
    } catch {
        Write-Log "GET $Url fehlgeschlagen: $($_.Exception.Message)" "DEBUG"
        return $null
    }
}

function Write-Diagnostics {
    Write-Log "Systeminformationen (fuer Fehlersuche, stehen im Log)" "STEP"
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($os) { Write-Log "Windows: $($os.Caption) $($os.Version), $env:PROCESSOR_ARCHITECTURE" }
    Write-Log "PowerShell: $($PSVersionTable.PSVersion)"
    Write-Log "Projekt: $Root"
    if ($os) { Write-Log ("RAM: {0:N1} GB" -f ($os.TotalVisibleMemorySize / 1MB)) }

    $drive = Get-PSDrive -Name ((Split-Path -Qualifier $Root).TrimEnd(":")) -ErrorAction SilentlyContinue
    if ($drive) {
        $free = $drive.Free / 1GB
        Write-Log ("Freier Speicher auf {0}: {1:N0} GB" -f $drive.Name, $free)
        if ($free -lt 30) {
            Write-Log "Weniger als 30 GB frei - Modelle und PyTorch brauchen ~20-25 GB" "WARN"
        }
    }

    # Proxies: the cause of "All connection attempts failed" in older Flay versions.
    foreach ($name in "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "OLLAMA_HOST") {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { Write-Log "Umgebung: $name=$value" }
    }
    $inet = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction SilentlyContinue
    if ($inet -and $inet.ProxyEnable -eq 1) {
        Write-Log "System-Proxy aktiv: $($inet.ProxyServer) (Ausnahmen: $($inet.ProxyOverride))" "WARN"
        Write-Log "Python sieht diesen Proxy auch fuer localhost; Flay umgeht ihn fuer lokale Server." "INFO"
    } else {
        Write-Log "Kein System-Proxy eingetragen"
    }
    if ($inet -and $inet.AutoConfigURL) { Write-Log "Proxy-Skript (PAC): $($inet.AutoConfigURL)" }

    $smi = Find-Program "nvidia-smi"
    if ($smi) {
        $gpu = (& $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null) -join "; "
        Write-Log "NVIDIA-GPU: $gpu"
        $driver = ($gpu -split ",")[1]
        if ($driver -and [version]$driver.Trim() -lt [version]"580.0") {
            Write-Log "NVIDIA-Treiber $($driver.Trim()) ist zu alt fuer PyTorch mit CUDA 13 (mind. 580) - Treiber aktualisieren, sonst laufen Stimme und Spracherkennung nur auf der CPU" "WARN"
        }
    } else {
        Write-Log "Keine NVIDIA-GPU gefunden (nvidia-smi fehlt) - Flay nimmt Piper statt Qwen3-TTS und rechnet auf der CPU" "WARN"
    }
}

function Install-Programs {
    Write-Log "Programme pruefen" "STEP"
    $ok = $true
    $ok = (Install-Program "git" "Git.Git" "Git") -and $ok
    $ok = (Install-Program "git-lfs" "GitHub.GitLFS" "Git LFS") -and $ok
    $ok = (Install-Program "uv" "astral-sh.uv" "uv") -and $ok
    $ok = (Install-Program "ollama" "Ollama.Ollama" "Ollama") -and $ok
    if (-not $ok) { throw "Es fehlen Programme (siehe oben)." }

    $version = (& (Find-Program "ollama") --version 2>&1) -join " "
    Write-Log "Ollama: $version"
    if ($version -match "(\d+\.\d+(\.\d+)?)") {
        if ([version]$Matches[1] -lt [version]"0.30") {
            Write-Log "Ollama $($Matches[1]) ist zu alt (mind. 0.30): winget upgrade Ollama.Ollama" "WARN"
        }
    }
}

function Update-Lfs {
    Write-Log "Mitgelieferte Stimmen (Git LFS)" "STEP"
    if (-not (Test-Path (Join-Path $Root ".git"))) {
        Write-Log "Kein Git-Checkout (ZIP?) - Git LFS uebersprungen" "WARN"
        return
    }
    Push-Location $Root
    try {
        Invoke-Logged "git" @("lfs", "install") -Quiet | Out-Null
        if ((Invoke-Logged "git" @("lfs", "pull")) -ne 0) { Write-Log "git lfs pull fehlgeschlagen" "WARN" }
    } finally {
        Pop-Location
    }
    $voice = Join-Path $Root "src\flay\tts\voices\ramona.wav"
    if ((Test-Path $voice) -and (Get-Item $voice).Length -lt 2048) {
        Write-Log "$voice ist nur ein LFS-Zeiger - Stimme fehlt (git lfs pull pruefen)" "ERROR"
    } else {
        Write-Log "Stimmen vorhanden" "OK"
    }
}

# Hyper-V, WSL 2 and Docker reserve port ranges; Ollama cannot listen inside one.
function Test-ReservedPort([int]$Port) {
    $output = (& netsh interface ipv4 show excludedportrange protocol=tcp 2>$null) -join "`n"
    Write-Log "Reservierte TCP-Ports (netsh):`n$output" "DEBUG"
    foreach ($match in [regex]::Matches($output, "(?m)^\s*(\d+)\s+(\d+)")) {
        $start = [int]$match.Groups[1].Value
        $end = [int]$match.Groups[2].Value
        if ($Port -ge $start -and $Port -le $end) {
            Write-Log "Port $Port ist von Windows reserviert ($start-$end, Hyper-V/WSL 2/Docker) - Ollama kann dort nicht lauschen. Abhilfe: Benutzer-Umgebungsvariable OLLAMA_HOST=127.0.0.1:11500 setzen, Ollama neu starten, in flay.yaml llm.base_url: http://127.0.0.1:11500/v1" "ERROR"
            return $true
        }
    }
    Write-Log "Port $Port ist nicht von Windows reserviert" "DEBUG"
    return $false
}

# Without Ollama in the autostart, Flay finds no LLM server after the next login.
function Test-OllamaAutostart {
    $startup = [Environment]::GetFolderPath("Startup")
    $entry = Get-ChildItem -Path $startup -Filter "Ollama*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($entry) {
        Write-Log "Ollama startet bei der Anmeldung ($($entry.FullName))" "OK"
    } else {
        Write-Log "Ollama steht nicht im Autostart ($startup) - nach einem Neustart erst Ollama starten, dann Flay" "WARN"
    }
}

function Start-OllamaServer {
    Write-Log "Ollama-Server" "STEP"
    Test-OllamaAutostart
    $body = Get-Direct "$OllamaUrl/api/version"
    if ($body) {
        Write-Log "Ollama laeuft auf ${OllamaUrl}: $body" "OK"
        return
    }
    $reserved = Test-ReservedPort 11434
    $running = Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue
    if ($running) {
        Write-Log ("Ollama-Prozesse laufen ({0}), der Server antwortet aber nicht" -f (($running | ForEach-Object { "$($_.ProcessName) PID $($_.Id)" }) -join ", ")) "WARN"
    }
    $listener = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        foreach ($l in $listener) {
            $process = Get-Process -Id $l.OwningProcess -ErrorAction SilentlyContinue
            Write-Log "Port 11434 belegt von $($process.ProcessName) (PID $($l.OwningProcess)) auf $($l.LocalAddress), antwortet aber nicht" "WARN"
        }
    }
    Write-Log "Ollama antwortet nicht - wird gestartet ..."
    $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $app) {
        Start-Process -FilePath $app
    } else {
        Start-Process -FilePath (Find-Program "ollama") -ArgumentList "serve" -WindowStyle Hidden
    }
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        $body = Get-Direct "$OllamaUrl/api/version" 1000
        if ($body) {
            Write-Log "Ollama laeuft: $body" "OK"
            return
        }
    }
    $serverLog = Join-Path $env:LOCALAPPDATA "Ollama\server.log"
    if (Test-Path $serverLog) {
        Write-Log "Letzte Zeilen aus $serverLog`:" "WARN"
        Get-Content $serverLog -Tail 30 | ForEach-Object { Write-Log "    $_" }
    }
    if ($reserved) { throw "Ollama kann Port 11434 nicht oeffnen (von Windows reserviert, siehe oben)." }
    throw "Ollama startet nicht ($OllamaUrl). Ollama-App von Hand starten und das Skript erneut ausfuehren."
}

function Install-Python {
    Write-Log "Python-Pakete (uv sync, beim ersten Mal mehrere GB fuer PyTorch)" "STEP"
    Push-Location $Root
    try {
        if ((Invoke-Logged "uv" @("sync")) -ne 0) { throw "uv sync fehlgeschlagen." }
        Invoke-Logged "uv" @("run", "python", "--version") -Quiet | Out-Null
        # What Python (and thus Flay's HTTP clients) see as proxies.
        Invoke-Logged "uv" @("run", "python", "-c",
            "import urllib.request; print('Python-Proxies:', urllib.request.getproxies() or 'keine')") | Out-Null
        Invoke-Logged "uv" @("run", "python", "-c",
            "import torch; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())") | Out-Null
        # The same connection Flay makes at startup (address from flay.yaml), with diagnosis.
        $check = "from flay.config import load_settings; from flay import net; " +
            "u = load_settings(None).llm.base_url; p = net.unreachable(u.rstrip('/') + '/models'); " +
            "print('LLM-Server aus Python erreichbar:', u) if p is None else print(net.llm_unreachable_message(u, p)); " +
            "raise SystemExit(p is not None)"
        if ((Invoke-Logged "uv" @("run", "python", "-c", $check)) -ne 0) {
            throw "Flay erreicht den LLM-Server nicht (Diagnose oben)."
        }
    } finally {
        Pop-Location
    }
    Write-Log "Python-Pakete installiert" "OK"
}

function Install-Models {
    Write-Log "Modelle laden (flay setup: Spracherkennung, Stimmen, LLM ueber Ollama)" "STEP"
    Push-Location $Root
    try {
        if ((Invoke-Logged "uv" @("run", "flay", "-v", "setup")) -ne 0) {
            throw "flay setup fehlgeschlagen (Details oben und im Log)."
        }
        Invoke-Logged "uv" @("run", "flay", "info") -Quiet | Out-Null
    } finally {
        Pop-Location
    }
    Write-Log "Modelle vollstaendig" "OK"
}

Write-Log "Flay-Setup gestartet, Log: $LogFile" "STEP"
try {
    Update-Path
    Write-Diagnostics
    Install-Programs
    Update-Lfs
    Start-OllamaServer
    Install-Python
    if ($SkipModels) { Write-Log "Modelle uebersprungen (-SkipModels)" "WARN" } else { Install-Models }
} catch {
    Write-Log $_.Exception.Message "ERROR"
    Write-Log $_.ScriptStackTrace "DEBUG"
    Write-Host "`nSetup fehlgeschlagen. Vollstaendiges Log: $LogFile" -ForegroundColor Red
    exit 1
}
Write-Host "`nFertig. Starten mit:  uv run flay run" -ForegroundColor Green
Write-Host "(In einem neuen Terminal, falls uv gerade erst installiert wurde.)"
Write-Host "Log: $LogFile"
exit 0
