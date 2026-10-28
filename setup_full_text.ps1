$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv-ocr\Scripts\python.exe')) {
    & '.\.venv\Scripts\python.exe' -m venv .venv-ocr
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated OCR environment.' }
}
& '.\.venv-ocr\Scripts\python.exe' -m pip install -r requirements-fulltext-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed. Existing environments were not changed.' }
& '.\.venv-ocr\Scripts\python.exe' -m visionbridge.prepare_full_text
if ($LASTEXITCODE -ne 0) { throw 'English OCR model verification failed.' }
Write-Output 'Full-text CPU environment ready. Existing CUDA and UI environments preserved.'
