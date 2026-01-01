
Write-Host "Setting up BitVoice alias..."

$functionDef = @"
function bitvoice {
    # Get current directory
    `$localDir = Get-Location
    # Run docker with volume mount
    docker run --rm -v "`$localDir`:/workspace" -w /workspace bitvoice:latest @args
}
"@

# Helper to add to profile
function Add-ToProfile {
    param([string]$Content)
    
    if (!(Test-Path $PROFILE)) {
        New-Item -Type File -Path $PROFILE -Force | Out-Null
    }
    
    $currentProfile = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
    if ($currentProfile -notmatch "function bitvoice") {
        Add-Content -Path $PROFILE -Value "`n# BitVoice Alias`n$Content"
        Write-Host "Added 'bitvoice' function to your PowerShell profile: $PROFILE"
    } else {
        Write-Host "BitVoice function already exists in your profile."
    }
}

# Add to current session
Invoke-Expression $functionDef
Write-Host "BitVoice is now available in this session."

# Ask to add to profile
$response = Read-Host "Do you want to add this alias to your PowerShell profile? (y/n)"
if ($response -eq 'y') {
    Add-ToProfile -Content $functionDef
    Write-Host "Please restart your terminal or run '. `$PROFILE' to make it permanent."
}
