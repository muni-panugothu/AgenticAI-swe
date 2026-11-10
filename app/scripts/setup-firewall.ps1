@param(
  [int]$Port = 8787
)

# Run PowerShell AS ADMINISTRATOR:
#   powershell -ExecutionPolicy Bypass -File app/scripts/setup-firewall.ps1

$ruleName = "AutoPilot AI Server $Port"

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
  Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule `
  -DisplayName $ruleName `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort $Port `
  -Profile Any `
  -Description "Allow Android phones on LAN to reach AutoPilot AI (Express) server" | Out-Null

Write-Host "OK: inbound TCP $Port allowed ($ruleName)"

$ip = (
  Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -match '^192\.168\.|^10\.' -and $_.PrefixOrigin -ne 'WellKnown' } |
  Select-Object -First 1 -ExpandProperty IPAddress
)

if ($ip) {
  Write-Host "Use in app/.env :"
  Write-Host "  EXPO_PUBLIC_API_URL=http://${ip}:${Port}"
} else {
  Write-Host "Could not detect LAN IP — run: ipconfig"
}
