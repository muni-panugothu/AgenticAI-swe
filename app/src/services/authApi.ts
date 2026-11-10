import { buildApiUrl } from './apiClient';
import {
  clearTokens,
  getAccessToken,
  getRefreshToken,
  saveTokens,
  StoredUser,
} from './authStorage';

export interface AuthUser extends StoredUser {}

export interface AuthSession {
  accessToken: string;
  refreshToken: string | null;
  user: AuthUser;
}

async function parseAuthError(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  const detail = (body as { detail?: { error?: string; code?: string }; error?: string }).detail;
  const message =
  detail?.error ?? (body as { error?: string }).error ?? `HTTP ${res.status}`;
  return new Error(message);
}

export async function register(
  email: string,
  password: string
): Promise<AuthSession> {
  const res = await fetch(buildApiUrl('/auth/register'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
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

export async function login(email: string, password: string): Promise<AuthSession> {
  const res = await fetch(buildApiUrl('/auth/login'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ email, password }),
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

export async function refreshAccessToken(): Promise<string | null> {
  const refresh = await getRefreshToken();
  const res = await fetch(buildApiUrl('/auth/refresh'), {
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
  let res = await fetch(buildApiUrl('/auth/me'), {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/json',
    },
    credentials: 'include',
  });
  if (res.status === 401) {
    const renewed = await refreshAccessToken();
    if (!renewed) return null;
    res = await fetch(buildApiUrl('/auth/me'), {
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
  if (token) {
    await fetch(buildApiUrl('/auth/logout'), {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
      },
      credentials: 'include',
    }).catch(() => undefined);
  }
  await clearTokens();
}
