# Run with phone connected via USB — forwards phone localhost:8787 → PC localhost:8787
adb reverse tcp:8787 tcp:8787
adb reverse --list
Write-Host "Done. Set app/.env to: EXPO_PUBLIC_API_URL=http://localhost:8787"
Write-Host "Then restart Metro: npx expo start --dev-client -c"
