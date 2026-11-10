# Downloads Gradle for Android builds when services.gradle.org is slow/blocked.
# Run from repo root: powershell -ExecutionPolicy Bypass -File app/scripts/setup-gradle.ps1

$ErrorActionPreference = "Stop"
$gradleVersion = "9.3.1"
$zipName = "gradle-$gradleVersion-bin.zip"
$wrapperDir = Join-Path $PSScriptRoot "..\android\gradle\wrapper" | Resolve-Path
$projectZip = Join-Path $wrapperDir $zipName
$url = "https://services.gradle.org/distributions/$zipName"

if (-not (Test-Path $projectZip)) {
  Write-Host "Downloading $url ..."
  curl.exe -L --connect-timeout 30 --max-time 600 -o $projectZip $url
  Write-Host "Saved to $projectZip"
} else {
  Write-Host "Gradle zip already exists: $projectZip"
}

# Update gradle-wrapper.properties to use local file (avoids Java network timeouts)
$propsFile = Join-Path $wrapperDir "gradle-wrapper.properties"
$localUrl = "file:///" + ($projectZip -replace '\\', '/')
$content = Get-Content $propsFile -Raw
$content = $content -replace 'distributionUrl=.*', "distributionUrl=$($localUrl -replace ':', '\:')"
if ($content -notmatch 'networkTimeout=') {
  $content = $content -replace '(distributionUrl=.*)', "`$1`nnetworkTimeout=600000"
}
if ($content -notmatch 'validateDistributionUrl=') {
  $content = $content + "`nvalidateDistributionUrl=false"
}
Set-Content $propsFile $content.TrimEnd()
Write-Host "Updated gradle-wrapper.properties to use local Gradle zip."
Write-Host "Now run: npm run android"
