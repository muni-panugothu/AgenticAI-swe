import { Platform } from 'react-native';
import Constants from 'expo-constants';

export const API_PORT = 8787;

function isPlaceholderUrl(url: string): boolean {
  const lower = url.toLowerCase();
  return (
    !url ||
    lower.includes('your_pc_ip') ||
    lower.includes('<db_') ||
    lower.includes('example.com')
  );
}

function normalizeBase(url: string): string {
  return url.trim().replace(/\/$/, '');
}

/**
 * Host Metro already uses for the JS bundle.
 * On a physical phone this is almost always the PC LAN IP (e.g. 192.168.x.x).
 */
export function getMetroHost(): string | null {
  const candidates = [
    Constants.expoConfig?.hostUri,
    (
      Constants as {
        manifest2?: { extra?: { expoClient?: { hostUri?: string } } };
      }
    ).manifest2?.extra?.expoClient?.hostUri,
    (
      Constants as {
        manifest?: { debuggerHost?: string; hostUri?: string };
      }
    ).manifest?.debuggerHost,
    (
      Constants as {
        manifest?: { debuggerHost?: string; hostUri?: string };
      }
    ).manifest?.hostUri,
  ];

  for (const hostUri of candidates) {
    if (!hostUri) continue;
    const host = hostUri
      .replace(/^[a-z+]+:\/\//i, '')
      .split('/')[0]
      .split(':')[0]
      .trim();

    if (
      host &&
      host !== 'localhost' &&
      host !== '127.0.0.1' &&
      host !== '10.0.2.2'
    ) {
      return host;
    }
  }

  return null;
}

/**
 * Build URL candidates for BOTH physical phones and emulators.
 *
 * CRITICAL: Do NOT gate on Constants.isDevice.
 * On many Expo dev-client builds, isDevice is false on real phones,
 * which previously forced ONLY http://10.0.2.2:8787 — that address
 * only works inside the Android Emulator, never on a physical device.
 */
export function getApiCandidates(): string[] {
  const seen = new Set<string>();
  const add = (url?: string | null) => {
    if (!url || isPlaceholderUrl(url)) return;
    const normalized = normalizeBase(url);
    if (!seen.has(normalized)) seen.add(normalized);
  };

  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  const metro = getMetroHost();

  // 1) Explicit env (highest priority when set correctly)
  add(fromEnv);

  // 2) Metro host = PC LAN IP the phone already uses for JS (best Wi‑Fi path)
  if (metro) {
    add(`http://${metro}:${API_PORT}`);
  }

  // 3) USB adb reverse: phone localhost → PC localhost:8787
  add(`http://127.0.0.1:${API_PORT}`);
  add(`http://localhost:${API_PORT}`);

  // 4) Emulator host alias LAST (never the only option)
  if (Platform.OS === 'android') {
    add(`http://10.0.2.2:${API_PORT}`);
  }

  return [...seen];
}

export function getDefaultApiBaseUrl(): string {
  const candidates = getApiCandidates();
  // Prefer anything that is NOT the emulator-only alias
  const preferred =
    candidates.find((u) => !u.includes('10.0.2.2')) ?? candidates[0];
  return preferred ?? `http://127.0.0.1:${API_PORT}`;
}

export function getDiagnostics(): Record<string, unknown> {
  return {
    platform: Platform.OS,
    isDevice: Constants.isDevice,
    metroHost: getMetroHost(),
    envUrl: process.env.EXPO_PUBLIC_API_URL ?? null,
    candidates: getApiCandidates(),
    expoHostUri: Constants.expoConfig?.hostUri ?? null,
  };
}
