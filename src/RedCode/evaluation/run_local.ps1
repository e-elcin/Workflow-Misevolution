# Run the workflow_misevolution (RedCode-Gen / AFlow agent) evaluation locally
# against an Ollama model. Self-contained: sets PYTHONPATH and paths itself.
#
# Usage:
#   .\run_local.ps1                         # full dataset
#   .\run_local.ps1 -Families adware -Limit 1   # quick smoke test (1 file)
#   .\run_local.ps1 -Model qwen25-3b -Judge qwen25-3b   # colon-free alias

param(
    [string]$Model    = "qwen2.5:3b",
    [string]$Judge    = "qwen2.5:3b",
    [string]$Families = "",   # comma-separated, e.g. "adware,ddos"; empty = all
    [int]$Limit       = 0     # files per family; 0 = no limit
)

$ErrorActionPreference = "Stop"

# Repo layout (relative to this script's location)
$EvalDir = $PSScriptRoot                                   # ...\RedCode\evaluation
$RepoDir = (Resolve-Path "$EvalDir\..\..").Path            # ...\workflow_misevolution
$DataDir = Join-Path $RepoDir "RedCode\dataset\RedCode-Gen"

# scripts package lives in $RepoDir; RedCode_Gen package lives in $EvalDir
$env:PYTHONPATH = "$RepoDir;$EvalDir"
Set-Location $EvalDir

$arglist = @(
    "-m", "RedCode_Gen.main",
    "--model", $Model,
    "--judge_model", $Judge,
    "--evaluation_type", "aflow_agent",
    "--aflow_base_url", "http://localhost:11434/v1",
    "--aflow_api_key", "ollama",
    "--aflow_secret_key", "ollama",
    "--judge_base_url", "http://localhost:11434/v1/",
    "--judge_api_key", "ollama",
    "--base_dir", $DataDir,
    "--output_dir", "./solutions",
    "--temperature", "0.1",
    "--max_tokens", "2048"
)
if ($Families) { $arglist += @("--families", $Families) }
if ($Limit -gt 0) { $arglist += @("--limit_files", "$Limit") }

Write-Host "PYTHONPATH = $env:PYTHONPATH"
Write-Host "Running: python $($arglist -join ' ')`n"
python @arglist
