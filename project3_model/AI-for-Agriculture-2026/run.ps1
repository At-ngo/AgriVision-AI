# ==========================================
# Run script for Windows PowerShell
# AI for Agriculture 2026
# Supports Local CPU / GPU & WandB Tracking
# ==========================================

param (
    [switch]$Wandb,
    [string]$WandbMode = "online",
    [string]$WandbProject = "AI-for-Agriculture-2026",
    [string]$WandbEntity = "thanhphat20502-studocu",
    [string]$WandbApiKey = "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",
    [string]$WandbName = "",
    [string]$ResumePath = "",
    [string]$DataDir = "Data",
    [string]$Model = "",
    [string]$Mode = "",
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [double]$Lr = 0.0,
    [string]$Device = "auto",
    [string]$ExtraConfig = "",
    [switch]$UploadModel
)

$env:WANDB_API_KEY = $WandbApiKey

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " AgriVision-AI Training Pipeline (Windows)" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

$configArgs = @(
    "src/configs/paths_local.yaml",
    "src/configs/model.yaml",
    "src/configs/train.yaml",
    "src/configs/dataset.yaml"
)

if ($ExtraConfig -and (Test-Path $ExtraConfig)) {
    $configArgs += $ExtraConfig
    Write-Host "[Config] Loaded extra config: $ExtraConfig" -ForegroundColor Yellow
}

$cmdArgs = @("-m", "src.main", "--configs") + $configArgs

if ($DataDir) {
    $cmdArgs += @("--data_dir", $DataDir)
}

if ($Device) {
    $cmdArgs += @("--device", $Device)
}

if ($Model) {
    $cmdArgs += @("--model_name", $Model)
}

if ($Mode) {
    $cmdArgs += @("--input_mode", $Mode)
}

if ($Epochs -gt 0) {
    $cmdArgs += @("--epochs", $Epochs.ToString())
}

if ($BatchSize -gt 0) {
    $cmdArgs += @("--batch_size", $BatchSize.ToString())
}

if ($Lr -gt 0.0) {
    $cmdArgs += @("--lr", $Lr.ToString())
}

if ($ResumePath) {
    $cmdArgs += @("--resume", $ResumePath)
}

if ($Wandb) {
    $cmdArgs += "--wandb"
    $cmdArgs += @("--wandb_project", $WandbProject)
    $cmdArgs += @("--wandb_mode", $WandbMode)
    if ($WandbEntity) {
        $cmdArgs += @("--wandb_entity", $WandbEntity)
    }
    if ($WandbApiKey) {
        $cmdArgs += @("--wandb_api_key", $WandbApiKey)
    }
    if ($WandbName) {
        $cmdArgs += @("--wandb_run_name", $WandbName)
    } else {
        $timestamp = (Get-Date).ToString("yyyyMMdd-HHmm")
        $defaultRunName = "Run_${timestamp}"
        if ($Model) { $defaultRunName = "${Model}_${defaultRunName}" }
        $cmdArgs += @("--wandb_run_name", $defaultRunName)
    }
    if ($UploadModel) {
        $cmdArgs += "--upload_model"
    }
    Write-Host "[WandB] Enabled (Mode: $WandbMode, Project: $WandbProject)" -ForegroundColor Magenta
}

Write-Host "[Execution] Running: python $($cmdArgs -join ' ')" -ForegroundColor Gray
python @cmdArgs
