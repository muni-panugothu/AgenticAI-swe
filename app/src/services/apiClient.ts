import {
  getApiCandidates,
  getDefaultApiBaseUrl,
  getDiagnostics,
  API_PORT,
} from '../config/api';

let activeBaseUrl = getDefaultApiBaseUrl();
let discoveryPromise: Promise<string> | null = null;
let lastProbeErrors: string[] = [];

export function getApiBaseUrl(): string {
  return activeBaseUrl;
}

export function setApiBaseUrl(url: string): void {
  activeBaseUrl = url.replace(/\/$/, '');
}

export function buildApiUrl(path: string): string {
  const base = activeBaseUrl.replace(/\/$/, '');
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${base}${suffix}`;
}

export function getLastProbeErrors(): string[] {
  return lastProbeErrors;
}

async function probeUrl(
  baseUrl: string,
  timeoutMs = 4000
): Promise<{ ok: boolean; detail: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const health = `${baseUrl.replace(/\/$/, '')}/health`;
  try {
    const res = await fetch(health, {
      method: 'GET',
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    if (res.ok) {
      return { ok: true, detail: `${baseUrl} → HTTP ${res.status}` };
    }
    return { ok: false, detail: `${baseUrl} → HTTP ${res.status}` };
  } catch (err) {
    const name = err instanceof Error ? err.name : 'Error';
    const message = err instanceof Error ? err.message : String(err);
    if (name === 'AbortError') {
      return { ok: false, detail: `${baseUrl} → timeout` };
    }
    return { ok: false, detail: `${baseUrl} → ${message}` };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Probe every candidate until one responds.
 * Never assume emulator vs physical device — try all paths.
 */
export async function discoverApiBaseUrl(): Promise<string> {
  if (discoveryPromise) return discoveryPromise;

  discoveryPromise = (async () => {
    const candidates = getApiCandidates();
    lastProbeErrors = [];

    if (__DEV__) {
      console.log('[api] discovery diagnostics', getDiagnostics());
    }

    for (const url of candidates) {
      const result = await probeUrl(url);
      if (result.ok) {
        activeBaseUrl = url;
        if (__DEV__) {
          console.log('[api] connected →', url);
        }
        return url;
      }
      lastProbeErrors.push(result.detail);
      if (__DEV__) {
        console.warn('[api] probe failed', result.detail);
      }
    }

    return activeBaseUrl;
  })();

  return discoveryPromise;
}

export async function checkServerHealth(): Promise<boolean> {
  await discoverApiBaseUrl();
  const result = await probeUrl(activeBaseUrl);
  return result.ok;
}

export function getConnectionHelpMessage(): string {
  const tried = getApiCandidates().join('\n• ');
  const errors =
    lastProbeErrors.length > 0
      ? `\n\nProbe results:\n• ${lastProbeErrors.join('\n• ')}`
      : '';
  const metro = getDiagnostics().metroHost;

  return (
    `Cannot reach AI server (port ${API_PORT}).\n\n` +
    `Tried:\n• ${tried}` +
    errors +
    `\n\nMetro host: ${metro ?? 'unknown'}` +
    `\nActive URL: ${activeBaseUrl}` +
    `\nisDevice: ${String(getDiagnostics().isDevice)}` +
    `\nenv: ${String(getDiagnostics().envUrl)}`
  );
}

export function resetApiDiscovery(): void {
  discoveryPromise = null;
  lastProbeErrors = [];
  activeBaseUrl = getDefaultApiBaseUrl();
}
