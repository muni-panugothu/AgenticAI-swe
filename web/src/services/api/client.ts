import { formatApiError } from '@/utils/errors';

/**
 * Empty base → same-origin requests (Vite proxies /auth, /api, /health).
 * Set VITE_API_BASE_URL only when the UI is served separately from the API.
 */
export function getApiBaseUrl(): string {
  const fromEnv = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (fromEnv === undefined || fromEnv.trim() === '') {
    return '';
  }
  return fromEnv.replace(/\/$/, '');
}

export function buildApiUrl(path: string): string {
  const base = getApiBaseUrl();
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${base}${suffix}`;
}

function networkErrorMessage(err: unknown): string {
  const detail = err instanceof Error ? err.message : String(err);
  if (/failed to fetch|networkerror|load failed/i.test(detail)) {
    return (
      'Cannot reach the API. Confirm the backend is running on port 8787, ' +
      'then reload this page. If you opened the site via a LAN IP, use the Vite ' +
      'dev server (proxy) or add that origin to backend CORS_ORIGINS.'
    );
  }
  return detail;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(buildApiUrl('/health'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    return res.ok;
  } catch {
    return false;
  }
}

type TokenGetter = () => string | null;
type RefreshFn = () => Promise<string | null>;

let getAccessToken: TokenGetter = () => null;
let refreshAccessToken: RefreshFn = async () => null;
let onAuthFailure: (() => void) | null = null;

export function configureAuthHandlers(options: {
  getAccessToken: TokenGetter;
  refreshAccessToken: RefreshFn;
  onAuthFailure?: () => void;
}): void {
  getAccessToken = options.getAccessToken;
  refreshAccessToken = options.refreshAccessToken;
  onAuthFailure = options.onAuthFailure ?? null;
}

async function parseError(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  return new Error(formatApiError(res.status, body, res.statusText));
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
  options: { auth?: boolean; retry?: boolean } = {}
): Promise<Response> {
  const { auth = true, retry = true } = options;
  const headers = new Headers(init.headers || {});
  if (!headers.has('Accept')) headers.set('Accept', 'application/json');

  if (auth) {
    const token = getAccessToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
  }

  let res: Response;
  try {
    res = await fetch(buildApiUrl(path), {
      ...init,
      headers,
      credentials: 'include',
    });
  } catch (err) {
    throw new Error(networkErrorMessage(err));
  }

  if (res.status !== 401 || !auth || !retry) {
    return res;
  }

  const next = await refreshAccessToken();
  if (!next) {
    onAuthFailure?.();
    return res;
  }

  headers.set('Authorization', `Bearer ${next}`);
  try {
    return await fetch(buildApiUrl(path), {
      ...init,
      headers,
      credentials: 'include',
    });
  } catch (err) {
    throw new Error(networkErrorMessage(err));
  }
}

export async function apiJson<T>(
  path: string,
  init: RequestInit = {},
  options?: { auth?: boolean; retry?: boolean }
): Promise<T> {
  const res = await apiFetch(path, init, options);
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}
