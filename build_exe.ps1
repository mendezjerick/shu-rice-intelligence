param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path build) {
        try { Remove-Item build -Recurse -Force -ErrorAction Stop }
        catch { Write-Warning "Could not delete 'build': $_" }
    }
    if (Test-Path dist) {
        try { Remove-Item dist -Recurse -Force -ErrorAction Stop }
        catch { Write-Warning "Could not delete 'dist' (close ShuRiceApp.exe if running): $_" }
    }
}

$pyinstaller = "pyinstaller"

function Add-DataArg($relativePath, $bundleTarget) {
    $src = Resolve-Path $relativePath
    return @("--add-data", "$($src.Path);$bundleTarget")
}

$dataArgs = @()
$dataArgs += Add-DataArg "rice.csv" "."
$dataArgs += Add-DataArg "backgrounds" "backgrounds"
$dataArgs += Add-DataArg "icons" "icons"
if (Test-Path "artifacts/best_model.joblib") {
    $dataArgs += Add-DataArg "artifacts/best_model.joblib" "artifacts"
}
if (Test-Path "artifacts/metrics.json") {
    $dataArgs += Add-DataArg "artifacts/metrics.json" "artifacts"
}
$dataArgs += Add-DataArg "src" "src"

$arguments = @(
    "--noconsole",
    "--onefile",
    "--name", "ShuRiceApp",
    "--icon", (Resolve-Path "icons/shu.ico").Path,
    "--collect-all", "PySide6",
    "--copy-metadata", "PySide6"
)
$arguments += $dataArgs
$arguments += "app.py"

Write-Host "Running: $pyinstaller $arguments"
& $pyinstaller @arguments
