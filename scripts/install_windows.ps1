
Write-Host "Setting up BitVoice alias..."

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop for Windows."
    exit 1
}

# Check if Docker daemon is running
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Docker daemon might not be running. Please start Docker Desktop."
}


Write-Host "Setting up BitVoice..."

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH. Please install Docker Desktop for Windows."
    exit 1
}

# Check if Docker daemon is running
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Docker daemon might not be running. Please start Docker Desktop."
}

# Create installation directory
$installDir = "$HOME\.bitvoice\bin"
if (!(Test-Path $installDir)) {
    try {
        New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    }
    catch {
        Write-Error "Failed to create installation directory $installDir : $_"
        exit 1
    }
}

# Create the batch file shim (works in CMD and PowerShell)
$batPath = "$installDir\bitvoice.bat"
$batContent = "@echo off
docker run --gpus all --rm -e PYTHONPATH=/app -v ""%cd%"":/workspace -w /workspace suryaanshrai515/bitvoice:latest %*
"
Set-Content -Path $batPath -Value $batContent

Write-Host "Created executable shim at: $batPath" -ForegroundColor Green

# Add to PATH Persistence
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$installDir*") {
    $newPath = "$userPath;$installDir"
    try {
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "Added $installDir to your User PATH." -ForegroundColor Green
        Write-Host "Please restart your terminal (or open a new one) to use 'bitvoice' system-wide." -ForegroundColor Cyan
    }
    catch {
        Write-Warning "Failed to update PATH environment variable: $_"
        Write-Warning "You can manually add '$installDir' to your Path."
    }
}
else {
    Write-Host "$installDir is already in your PATH." -ForegroundColor Green
    Write-Host "If 'bitvoice' is not working, try restarting your terminal." -ForegroundColor Cyan
}

# Add ephemeral alias for current session so it works immediately
function bitvoice {
    # Get current directory
    $localDir = Get-Location
    # Run docker with volume mount
    docker run --gpus all --rm -e PYTHONPATH=/app -v "${localDir}:/workspace" -w /workspace suryaanshrai515/bitvoice:latest @args
}
Write-Host "BitVoice is available in this session." -ForegroundColor Green
