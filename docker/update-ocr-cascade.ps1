# Builds ocr-cascade-webhook from the current working tree on the server's Docker
# daemon (over SSH - the PC needs only the docker CLI, no daemon), tags it with the
# short git sha (traceable, and never collides with a previous build the way :latest
# does), pins that tag in the stack's .env on the server, and recreates ONLY that one
# container via --no-deps - the other services of the Paperless stack are left
# untouched.
#
# Where to deploy: -SshHost / -StackDir, or PAPERFLOW_DEPLOY_HOST /
# PAPERFLOW_DEPLOY_STACK_DIR from the environment or this repository's .env.
#
# Usage (from anywhere):
#   pwsh <repo>\docker\update-ocr-cascade.ps1 [-SshHost <host>] [-StackDir <dir>]
#
# Rollback: the previous tag is printed before the switch, and the stack's .env is
# copied to .env.bak-YYYYMMDD (first deploy of the day). Put an older sha (any tag
# still present in `docker images ocr-cascade-webhook` on the server) back into
# OCR_CASCADE_WEBHOOK_VERSION and run `docker compose up -d --no-deps
# ocr-cascade-webhook` in the stack folder.

param(
    [string]$SshHost = $env:PAPERFLOW_DEPLOY_HOST,
    [string]$StackDir = $env:PAPERFLOW_DEPLOY_STACK_DIR
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent

function Get-DotEnvValue([string]$Name) {
    $file = Join-Path $repoRoot ".env"
    if (-not (Test-Path $file)) { return $null }
    $line = Get-Content $file | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -Last 1
    if ($line) { return ($line -split '=', 2)[1].Trim().Trim('"') }
    return $null
}

if (-not $SshHost) { $SshHost = Get-DotEnvValue "PAPERFLOW_DEPLOY_HOST" }
if (-not $StackDir) { $StackDir = Get-DotEnvValue "PAPERFLOW_DEPLOY_STACK_DIR" }
if (-not $SshHost) { throw "PAPERFLOW_DEPLOY_HOST is not set (pass -SshHost or add it to .env)" }
if (-not $StackDir) { throw "PAPERFLOW_DEPLOY_STACK_DIR is not set (pass -StackDir or add it to .env)" }

$sha = (git -C $repoRoot rev-parse --short HEAD).Trim()
$dirty = git -C $repoRoot status --porcelain
if ($dirty) {
    $sha = "$sha-dirty"
    Write-Warning "Working tree has uncommitted changes - tagging as $sha"
}

docker -H "ssh://$SshHost" build -f "$repoRoot\docker\service.Dockerfile" -t "ocr-cascade-webhook:$sha" $repoRoot
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

$remote = @"
set -e
cd '$StackDir'
echo "previous: `$(grep '^OCR_CASCADE_WEBHOOK_VERSION=' .env)"
cp -n .env ".env.bak-`$(date +%Y%m%d)"
sed -i 's/^OCR_CASCADE_WEBHOOK_VERSION=.*/OCR_CASCADE_WEBHOOK_VERSION=$sha/' .env
docker compose up -d --no-deps ocr-cascade-webhook
"@
# PowerShell appends CRLF to text piped into a native command, so the CR is
# stripped on the server - otherwise it sticks to the last word of the script.
$remote | ssh $SshHost "tr -d '\r' | sh -s"
if ($LASTEXITCODE -ne 0) { throw "deploy on $SshHost failed" }

Write-Host "Deployed ocr-cascade-webhook:$sha to $SshHost" -ForegroundColor Green
