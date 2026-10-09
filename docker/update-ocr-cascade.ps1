# Builds ocr-cascade-webhook from the current working tree, tags it with the short
# git sha (traceable, and never collides with a previous build the way :latest does),
# pins that tag in D:\Projects\YAML\paperless-ngx\.env, and recreates ONLY that one
# container via --no-deps - the other three services in that compose file
# (paperless-broker, paperless-db, paperless-webserver) are left untouched.
#
# Usage (from anywhere):
#   pwsh D:\Projects\git\repos\paperflow\docker\update-ocr-cascade.ps1
#
# Rollback: edit OCR_CASCADE_WEBHOOK_VERSION in D:\Projects\YAML\paperless-ngx\.env
# back to an older sha (any tag still present in `docker images ocr-cascade-webhook`),
# then rerun the `docker compose up -d --no-deps` line at the bottom by hand.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$composeDir = "D:\Projects\YAML\paperless-ngx"
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
