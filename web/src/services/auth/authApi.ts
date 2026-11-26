import { apiFetch, apiJson, buildApiUrl } from '@/services/api/client';
import {
  clearSession,
  getRefreshToken,
  saveSession,
} from '@/services/auth/tokenStore';
import type { AuthTokensResponse, AuthUser } from '@/types/api';
import { extractErrorBody, formatApiError, formatValidationDetails } from '@/utils/errors';

export interface RegisterInput {
  displayName: string;
  email: string;
  password: string;
  acceptedTerms: true;
  acceptedPrivacy: true;
}

export interface LoginInput {
  email: string;
  password: string;
  rememberMe?: boolean;
}

async function parseAuthError(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  const extracted = extractErrorBody(body);
  const details = formatValidationDetails(
    extracted.details ?? (body as { details?: unknown }).details
  );
  return new Error(
    formatApiError(res.status, body, res.statusText) ||
      details ||
      `HTTP ${res.status}`
  );
}

function persist(tokens: AuthTokensResponse): AuthUser {
  saveSession({
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token ?? null,
    user: tokens.user,
  });
  return tokens.user;
}

function wrapNetwork<T>(promise: Promise<T>): Promise<T> {
  return promise.catch((err: unknown) => {
    const detail = err instanceof Error ? err.message : String(err);
    if (/failed to fetch|networkerror|load failed/i.test(detail)) {
      throw new Error(
        'Cannot reach the API. Confirm the backend is running on port 8787, then reload this page.'
      );
    }
    throw err instanceof Error ? err : new Error(detail);
  });
}

export async function register(input: RegisterInput): Promise<AuthUser> {
  const res = await wrapNetwork(
    fetch(buildApiUrl('/auth/register'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        display_name: input.displayName.trim(),
        email: input.email.trim().toLowerCase(),
        password: input.password,
        accepted_terms: true,
        accepted_privacy: true,
      }),
    })
  );
  if (!res.ok) throw await parseAuthError(res);
  return persist((await res.json()) as AuthTokensResponse);
}

export async function login(input: LoginInput): Promise<AuthUser> {
  const res = await wrapNetwork(
    fetch(buildApiUrl('/auth/login'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        email: input.email.trim().toLowerCase(),
        password: input.password,
        remember_me: Boolean(input.rememberMe),
      }),
    })
  );
  if (!res.ok) throw await parseAuthError(res);
  return persist((await res.json()) as AuthTokensResponse);
}

export async function refreshSession(): Promise<string | null> {
  const bodyRefresh = getRefreshToken();
  let res: Response;
  try {
    res = await fetch(buildApiUrl('/auth/refresh'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      credentials: 'include',
      body: JSON.stringify(
        bodyRefresh ? { refresh_token: bodyRefresh } : {}
      ),
    });
  } catch {
    clearSession();
    return null;
  }
  if (!res.ok) {
    clearSession();
    return null;
  }
  const data = (await res.json()) as AuthTokensResponse;
  persist(data);
  return data.access_token;
}

export async function logout(): Promise<void> {
  const refresh = getRefreshToken();
  try {
    await apiFetch(
      '/auth/logout',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(refresh ? { refresh_token: refresh } : {}),
      },
      { auth: true, retry: false }
    );
  } catch {
    // Always clear local state.
  } finally {
    clearSession();
  }
}

export async function fetchMe(): Promise<AuthUser> {
  return apiJson<AuthUser>('/auth/me', { method: 'GET' });
}

export async function fetchLegalVersions(): Promise<{
  terms_version: string;
  privacy_version: string;
}> {
  return apiJson('/auth/legal-versions', { method: 'GET' }, { auth: false });
}
