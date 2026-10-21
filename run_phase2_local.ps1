param(
    [ValidateSet('Auto','Pilot','PrepareOnly','Resume')]
    [string]$Mode = 'Auto',
    [string]$RunName = '',
    [string]$ValidationZip = ''
)
$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Project
$Python = Join-Path $Project '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "GPU environment missing: $Python" }
$Manifest = Join-Path $Project 'reports\phase2_local_dev_evaluation\drive_inputs\split_manifest.json'
$Rules = Join-Path $Project 'reports\phase2_local_dev_evaluation\drive_inputs\official_normalization.json'
if ($Mode -ne 'Resume') {
    if (-not (Test-Path -LiteralPath $Manifest)) { throw "Missing exact run02 manifest: $Manifest" }
    if (-not (Test-Path -LiteralPath $Rules)) { throw "Missing run02 Phase 1 scoring rules: $Rules" }
    $RunName = 'run-local-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
}
if ($RunName -notmatch '^run-local-[A-Za-z0-9-]+$') { throw 'RunName must be the printed run-local identifier.' }
$RunRoot = Join-Path $Project "reports\phase2_local_dev_evaluation\$RunName"
$StatePath = Join-Path $RunRoot 'progress.json'
$EnvironmentPath = Join-Path $RunRoot 'environment.json'
if ($Mode -ne 'Resume') {
    if (Test-Path -LiteralPath $RunRoot) { throw "Run folder exists and will be preserved: $RunRoot" }
    New-Item -ItemType Directory -Path $RunRoot | Out-Null
    $env:VISIONBRIDGE_DEVICE = 'cuda'
    $env:VISIONBRIDGE_VQA_MODE = 'v2'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:128'
    $prepareArgs = @('-B','prepare_phase2_local.py','--manifest',$Manifest,'--rules',$Rules,'--run-dir',$RunRoot)
    if ($ValidationZip) { $prepareArgs += @('--archive',$ValidationZip) }
    Write-Host "[$RunName] Checking .venv-gpu, CUDA, BLIP v2, manifest, and image archive."
    $oldPreference = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Python @prepareArgs 2>&1 | Tee-Object -FilePath (Join-Path $RunRoot 'prepare.log') | Out-Host
    $pythonExit = $LASTEXITCODE; $ErrorActionPreference = $oldPreference
    if ($pythonExit -ne 0) { throw "Preparation failed (exit $pythonExit). Log: $RunRoot\prepare.log" }
    $prepared = Get-Content -Raw (Join-Path $RunRoot 'prepared.json') | ConvertFrom-Json
    $envState = [ordered]@{ run_name=$RunName; device='cuda'; vqa_weight_dtype='float32'; phase1_checkpoint_hashes=$prepared.checkpoint_hashes; resume_stage='pilot' }
    $envState | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $EnvironmentPath
    $progress = [ordered]@{ run_name=$RunName; prepared=$true; pilot_verified=$false; full_complete=$false; active_profile='float32'; updated=(Get-Date -Format o) }
    $progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
} else {
    if (-not (Test-Path -LiteralPath $RunRoot) -or -not (Test-Path -LiteralPath $StatePath)) { throw "No resumable local run named $RunName." }
    $progress = Get-Content -Raw $StatePath | ConvertFrom-Json
    if ($progress.run_name -ne $RunName -or -not $progress.prepared) { throw 'Run state does not match its folder; preserving artifacts.' }
    $envState = Get-Content -Raw $EnvironmentPath | ConvertFrom-Json
    if ($progress.full_complete) { Write-Host "Run already completed: $RunRoot\report\REPORT.md"; exit 0 }
    $env:VISIONBRIDGE_DEVICE = 'cuda'; $env:VISIONBRIDGE_VQA_MODE='v2'; $env:CUDA_VISIBLE_DEVICES='0'
    $env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
}
if ($Mode -eq 'PrepareOnly') { Write-Host "Prepared: $RunRoot"; Write-Host "To run pilot then full dev: .\run_phase2_local.ps1 -Mode Resume -RunName $RunName"; exit 0 }

$V2 = Join-Path $Project 'models\blip_vizwiz_v2'
$Dev = Join-Path $RunRoot 'inputs\run02-dev-only.json'
$Images = Join-Path $RunRoot 'data\images'
$Checkpoint = Join-Path $RunRoot 'evaluation'
$ModelCache = Join-Path $RunRoot 'model-cache'
$EvidenceCache = Join-Path $RunRoot 'evidence-cache'
$LogRoot = Join-Path $RunRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogRoot,$ModelCache,$EvidenceCache | Out-Null
if (-not (Test-Path -LiteralPath $V2)) { throw "Improved BLIP v2 checkpoint missing: $V2" }
if (-not (Test-Path -LiteralPath $Dev) -or -not (Test-Path -LiteralPath $Images)) { throw 'Prepared dev input/images missing. Preserve the run; inspect prepare.log.' }
$oldPreference = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
& $Python -B prepare_easyocr_models.py --directory (Join-Path $ModelCache 'easyocr') 2>&1 | Tee-Object -FilePath (Join-Path $RunRoot 'easyocr-models.log') | Out-Host
$pythonExit = $LASTEXITCODE; $ErrorActionPreference = $oldPreference
if ($pythonExit -ne 0) { throw "EasyOCR model preparation failed (exit $pythonExit). Log: $RunRoot\easyocr-models.log" }

function Invoke-Phase2Stage([string]$Stage,[string]$Profile,[switch]$ResumeStage) {
    $dest = Join-Path $Checkpoint "$Profile\$Stage"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    $log = Join-Path $LogRoot "$Profile-$Stage.log"
    if ($Profile -eq 'fp16') { $env:VISIONBRIDGE_LOW_MEMORY_VQA='1' } else { Remove-Item Env:\VISIONBRIDGE_LOW_MEMORY_VQA -ErrorAction SilentlyContinue }
    $args = @('-B','evaluate_phase2_dev.py','--dev-manifest',$Dev,'--images',$Images,'--checkpoint',$V2,
              '--output',$dest,'--evidence-cache',$EvidenceCache,'--model-cache',$ModelCache,
              '--yolo-model',(Join-Path $ModelCache 'yolov8n.pt'),'--device','cuda','--mode',$Stage)
    if ($Stage -eq 'pilot') { $args += @('--limit','20') }
    if ($ResumeStage) { $args += '--resume' }
    Write-Host "[$RunName] $Stage, CUDA BLIP ($Profile weights), CPU OCR, batch size 1. Progress is printed per example."
    $oldPreference = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    & $Python @args 2>&1 | Tee-Object -FilePath $log | Out-Host
    $pythonExit = $LASTEXITCODE; $ErrorActionPreference = $oldPreference
    if ($pythonExit -ne 0) {
        $tail = Get-Content -LiteralPath $log -Tail 100 -ErrorAction SilentlyContinue | Out-String
        throw "Stage $Stage failed (exit $pythonExit). Log: $log`n$tail"
    }
    return $dest
}

function Test-CudaOutOfMemory([string]$Message) {
    return $Message -match '(?i)CUDA out of memory|outofmemoryerror|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory'
}

function Verify-Pilot([string]$PilotPath) {
    $result = $PilotPath
    if ((Split-Path -Leaf $result) -notlike 'comparison-*') {
        $result = Get-ChildItem -Directory $PilotPath -Filter 'comparison-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $result) { throw "Pilot comparison output was not created: $PilotPath" }
    $args = @('-B','verify_phase2_pilot.py','--run-dir',$RunRoot,'--result-dir',$result)
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Pilot integrity check failed. Preserve all results: $PilotPath" }
}

if ($Mode -eq 'Resume' -and $progress.active_profile -eq 'fp16') { $profile='fp16' } else { $profile='fp32' }
$pilotDir = Join-Path $Checkpoint "$profile\pilot"
$fullDir = Join-Path $Checkpoint "$profile\full"
try {
    if (Test-Path -LiteralPath $fullDir) {
        if ($Mode -ne 'Resume') { throw "Full evaluation folder exists; preserving it: $fullDir" }
        $null = Invoke-Phase2Stage 'full' $profile -ResumeStage
    } else {
        if (Test-Path -LiteralPath $pilotDir) {
            if ($Mode -ne 'Resume') { throw "Pilot folder exists; preserve it and use -Mode Resume: $pilotDir" }
            $pilotPath = Invoke-Phase2Stage 'pilot' $profile -ResumeStage
        } else {
            $pilotPath = Invoke-Phase2Stage 'pilot' $profile
        }
        Verify-Pilot $pilotPath
        $progress.pilot_verified=$true; $progress.active_profile=$profile; $progress.updated=(Get-Date -Format o)
        $progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
        if ($Mode -eq 'Pilot') { Write-Host "Pilot integrity checks passed. Full evaluation not requested. Run name: $RunName"; exit 0 }
        try { $fullPath = Invoke-Phase2Stage 'full' $profile }
        catch {
            if ($profile -eq 'fp16' -or -not (Test-CudaOutOfMemory $_.ToString())) { throw }
            Write-Warning 'CUDA ran out of memory. Preserving the FP32 partial run and retrying the full development set with FP16 BLIP weights; OCR stays on CPU. This starts a fresh model-prediction folder and reuses the OCR cache.'
            $profile='fp16'; $envState.vqa_weight_dtype='float16'; $progress.active_profile='fp16'
            $envState | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $EnvironmentPath
            $progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
            $fullPath = Invoke-Phase2Stage 'full' $profile
        }
    }
} catch {
    if ($profile -eq 'fp32' -and (Test-CudaOutOfMemory $_.ToString())) {
        Write-Warning 'CUDA ran out of memory. Preserving the interrupted FP32 pilot and starting a new FP16-weight pilot. Existing extraction cache will be reused.'
        $profile='fp16'; $envState.vqa_weight_dtype='float16'; $progress.active_profile='fp16'
        $envState | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $EnvironmentPath
        $progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
        $pilotPath = Invoke-Phase2Stage 'pilot' $profile
        Verify-Pilot $pilotPath
        $progress.pilot_verified=$true; $progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
        $fullPath = Invoke-Phase2Stage 'full' $profile
    } else { throw }
}
if ($Mode -eq 'Pilot') { Write-Host "Pilot integrity checks passed. Full evaluation not requested. Run name: $RunName"; exit 0 }
$reportDir = Join-Path $RunRoot 'report'
& $Python -B build_phase2_local_report.py --run-dir $RunRoot --profile $profile
if ($LASTEXITCODE -ne 0) { throw "Report generation failed; evaluation files are preserved in $RunRoot" }
$progress.full_complete=$true; $progress.active_profile=$profile; $progress.updated=(Get-Date -Format o)
$progress | ConvertTo-Json | Set-Content -Encoding UTF8 $StatePath
Write-Host "Completed. Report: $reportDir\REPORT.md"
