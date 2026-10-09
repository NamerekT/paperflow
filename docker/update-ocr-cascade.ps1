# Builds ocr-cascade-webhook from the current working tree, tags it with the short
# git sha (traceable, and never collides with a previous build the way :latest does),
# pins that tag in the Paperless compose project's .env, and recreates ONLY that one
# container via --no-deps - the other three services in that compose file
# (paperless-broker, paperless-db, paperless-webserver) are left untouched.
#
# The compose project lives outside this repository: -ComposeDir, or
# PAPERLESS_COMPOSE_DIR from the environment or this repository's .env.
#
# Usage (from anywhere):
#   pwsh <repo>\docker\update-ocr-cascade.ps1 [-ComposeDir <dir>]
#
# Rollback: edit OCR_CASCADE_WEBHOOK_VERSION in the compose project's .env
# back to an older sha (any tag still present in `docker images ocr-cascade-webhook`),
# then rerun the `docker compose up -d --no-deps` line at the bottom by hand.

param([string]$ComposeDir = $env:PAPERLESS_COMPOSE_DIR)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent

if (-not $ComposeDir) {
    $repoEnv = Join-Path $repoRoot ".env"
    if (Test-Path $repoEnv) {
        $line = Get-Content $repoEnv | Where-Object { $_ -match '^\s*PAPERLESS_COMPOSE_DIR\s*=' } |
            Select-Object -Last 1
        if ($line) { $ComposeDir = ($line -split '=', 2)[1].Trim().Trim('"') }
    }
}
if (-not $ComposeDir) { throw "PAPERLESS_COMPOSE_DIR is not set (pass -ComposeDir or add it to .env)" }
if (-not [System.IO.Path]::IsPathRooted($ComposeDir)) { $ComposeDir = Join-Path $repoRoot $ComposeDir }

$composeDir = (Resolve-Path $ComposeDir).Path
$envFile = "$composeDir\.env"

$sha = (git -C $repoRoot rev-parse --short HEAD).Trim()
$dirty = git -C $repoRoot status --porcelain
if ($dirty) {
    $sha = "$sha-dirty"
    Write-Warning "Working tree has uncommitted changes - tagging as $sha"
}

docker build -f "$repoRoot\docker\service.Dockerfile" -t "ocr-cascade-webhook:$sha" $repoRoot
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

(Get-Content $envFile) -replace 'OCR_CASCADE_WEBHOOK_VERSION=.*', "OCR_CASCADE_WEBHOOK_VERSION=$sha" |
    Set-Content $envFile

docker compose -f "$composeDir\docker-compose.yml" up -d --no-deps ocr-cascade-webhook
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

Write-Host "Deployed ocr-cascade-webhook:$sha" -ForegroundColor Green
