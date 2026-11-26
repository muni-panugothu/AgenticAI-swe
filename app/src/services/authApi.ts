import { buildApiUrl, discoverApiBaseUrl } from './apiClient';
import {
  clearTokens,
  getAccessToken,
  getRefreshToken,
  saveRememberedEmail,
  saveTokens,
  StoredUser,
} from './authStorage';

export interface AuthUser extends StoredUser {}

export interface AuthSession {
  accessToken: string;
  refreshToken: string | null;
  user: AuthUser;
}

export interface RegisterPayload {
  displayName: string;
  email: string;
  password: string;
  acceptedTerms: boolean;
  acceptedPrivacy: boolean;
}

function formatValidationDetails(details: unknown): string | null {
  if (!Array.isArray(details) || details.length === 0) return null;
  const parts: string[] = [];
  for (const item of details) {
    if (!item || typeof item !== 'object') continue;
    const row = item as { msg?: string; loc?: unknown[] };
    const msg = typeof row.msg === 'string' ? row.msg : null;
    if (!msg) continue;
    // Drop noisy pydantic prefixes
    const clean = msg.replace(/^Value error,\s*/i, '').trim();
    if (clean && !parts.includes(clean)) parts.push(clean);
  }
  return parts.length ? parts.join('. ') : null;
}

async function parseAuthError(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  const detail = (body as {
    detail?: { error?: string; code?: string } | string;
    error?: string;
    details?: unknown;
  }).detail;

  if (typeof detail === 'string') {
    return new Error(detail);
  }

  const fromDetails = formatValidationDetails(
    (body as { details?: unknown }).details ??
      (typeof detail === 'object' && detail && 'details' in detail
        ? (detail as { details?: unknown }).details
        : null)
  );

  const message =
    detail?.error ??
    (body as { error?: string }).error ??
    fromDetails ??
    `HTTP ${res.status}`;
  return new Error(message);
}

async function readyFetch(
  path: string,
  init: RequestInit
): Promise<Response> {
  try {
    await discoverApiBaseUrl();
  } catch {
    // Fall through — still attempt the request with current base URL.
  }
  return fetch(buildApiUrl(path), init);
}

export async function register(payload: RegisterPayload): Promise<AuthSession> {
  const res = await readyFetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({
      display_name: payload.displayName.trim(),
      email: payload.email.trim().toLowerCase(),
      password: payload.password,
      accepted_terms: true,
      accepted_privacy: true,
    }),
  });
  if (!res.ok) throw await parseAuthError(res);
  const data = await res.json();
  const session: AuthSession = {
    accessToken: data.access_token,
    refreshToken: data.refresh_token ?? null,
    user: data.user,
  };
  await saveTokens(session);
  return session;
}

export async function login(
  email: string,
  password: string,
  rememberMe = false
): Promise<AuthSession> {
  const res = await readyFetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({
      email: email.trim().toLowerCase(),
      password,
      remember_me: rememberMe,
    }),
  });
  if (!res.ok) throw await parseAuthError(res);
  const data = await res.json();
  const session: AuthSession = {
    accessToken: data.access_token,
    refreshToken: data.refresh_token ?? null,
    user: data.user,
  };
  await saveTokens(session);
  await saveRememberedEmail(rememberMe ? email.trim().toLowerCase() : null);
  return session;
}

export async function requestPasswordReset(email: string): Promise<string> {
  const res = await readyFetch('/auth/forgot-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email: email.trim().toLowerCase() }),
  });
  if (!res.ok) throw await parseAuthError(res);
  const data = await res.json();
  return (
    data.message ||
    'If an account exists for that email, a password reset link has been sent.'
  );
}

export async function resetPassword(payload: {
  token: string;
  password: string;
  confirmPassword: string;
}): Promise<string> {
  const res = await readyFetch('/auth/reset-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({
      token: payload.token.trim(),
      password: payload.password,
      confirm_password: payload.confirmPassword,
    }),
  });
  if (!res.ok) throw await parseAuthError(res);
  const data = await res.json();
  return data.message || 'Password updated. You can sign in with your new password.';
}

export async function refreshAccessToken(): Promise<string | null> {
  const refresh = await getRefreshToken();
  const res = await readyFetch('/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify(refresh ? { refresh_token: refresh } : {}),
  });
  if (!res.ok) {
    await clearTokens();
    return null;
  }
  const data = await res.json();
  await saveTokens({
    accessToken: data.access_token,
    refreshToken: data.refresh_token ?? refresh,
    user: data.user,
  });
  return data.access_token as string;
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const token = await getAccessToken();
  if (!token) return null;
  let res = await readyFetch('/auth/me', {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/json',
    },
    credentials: 'include',
  });
  if (res.status === 401) {
    const renewed = await refreshAccessToken();
    if (!renewed) return null;
    res = await readyFetch('/auth/me', {
      headers: {
        Authorization: `Bearer ${renewed}`,
        Accept: 'application/json',
      },
      credentials: 'include',
    });
  }
  if (!res.ok) return null;
  return res.json() as Promise<AuthUser>;
}

export async function logout(): Promise<void> {
  const token = await getAccessToken();
  const refresh = await getRefreshToken();
  try {
    await readyFetch('/auth/logout', {
      method: 'POST',
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(refresh ? { refresh_token: refresh } : {}),
    });
  } catch {
    // Always clear local session even if the network call fails.
  }
  await clearTokens();
}
