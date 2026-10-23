param([switch]$SkipDependencies)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup command failed: $Exe $Arguments" }
}
Invoke-Checked 'git' @('lfs','version')
# Fetch only approved deployment LFS files, not the historical BLIP checkpoint.
Invoke-Checked 'git' @('lfs','pull','--include=models/blip_vizwiz_v2/model.safetensors,runtime/classifier/heads.joblib')
if (-not $SkipDependencies) {
    foreach ($EnvironmentName in @('.venv','.venv-gpu')) {
        if (-not (Test-Path -LiteralPath "$EnvironmentName\Scripts\python.exe")) {
            Invoke-Checked 'py' @('-3.11','-m','venv',$EnvironmentName)
        }
    }
    Invoke-Checked '.\.venv\Scripts\python.exe' @('-m','pip','install','-r','requirements-final-ui.txt')
    Invoke-Checked '.\.venv-gpu\Scripts\python.exe' @('-m','pip','install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121')
    Invoke-Checked '.\.venv-gpu\Scripts\python.exe' @('-m','pip','install','-r','requirements-final-gpu.txt')
    & '.\setup_full_text.ps1'
    if ($LASTEXITCODE -ne 0) { throw 'RapidOCR setup failed' }
}
Invoke-Checked '.\.venv-gpu\Scripts\python.exe' @('-c',"import torch; assert torch.__version__ == '2.5.1+cu121'; assert torch.cuda.is_available(), 'NVIDIA CUDA is required; no CPU VQA fallback'; print(torch.cuda.get_device_name(0))")
Invoke-Checked '.\.venv\Scripts\python.exe' @('public_models.py','download')
foreach ($EnvironmentName in @('.venv','.venv-gpu','.venv-ocr')) {
    Invoke-Checked ".\$EnvironmentName\Scripts\python.exe" @('-m','pip','check')
}
Write-Output 'Runtime assets verified. Launch: .\.venv\Scripts\python.exe launch_workspace.py --port 7865'
Write-Output 'Initial downloads require Internet. Speech playback/microphone require a manual browser check.'
