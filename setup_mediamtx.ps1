$ErrorActionPreference = "Stop"

$Version = "1.12.2"

switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { $Arch = "amd64" }
    "ARM64" { $Arch = "arm64" }
    default {
        Write-Error "Unsupported architecture: $env:PROCESSOR_ARCHITECTURE"
    }
}

$Archive = "mediamtx_v$($Version)_windows_$($Arch).zip"
$Url = "https://github.com/bluenviron/mediamtx/releases/download/v$($Version)/$Archive"

Write-Host "Downloading mediamtx v$Version for windows/$Arch..."
Invoke-WebRequest -Uri $Url -OutFile $Archive

Write-Host "Extracting..."
Expand-Archive -Path $Archive -DestinationPath . -Force
Remove-Item $Archive

Write-Host ""
Write-Host "Done! Run mediamtx with:"
Write-Host "  .\mediamtx.exe"
Write-Host ""
Write-Host "Then start camserver with RTSP enabled:"
Write-Host "  python camserver.py rtsp"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Note: ffmpeg is also required for RTSP mode."
    Write-Host "Install it with: winget install Gyan.FFmpeg"
}