import type { AuthUser } from '@/types/api';

const ACCESS_KEY = 'agentic.web.access_token';
const REFRESH_KEY = 'agentic.web.refresh_token';
const USER_KEY = 'agentic.web.user';

/**
 * Access tokens are kept in sessionStorage (cleared when the tab closes).
 * Refresh tokens may also arrive via HttpOnly cookie from the backend.
 * sessionStorage is NOT equivalent to HttpOnly cookie protection — documented
 * limitation for SPA + Bearer API compatibility with the Expo client contract.
 */
export function getAccessToken(): string | null {
  return sessionStorage.getItem(ACCESS_KEY);
}

export function getRefreshToken(): string | null {
  return sessionStorage.getItem(REFRESH_KEY);
}

export function getStoredUser(): AuthUser | null {
  const raw = sessionStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

export function saveSession(input: {
  accessToken: string;
  refreshToken?: string | null;
  user: AuthUser;
}): void {
  sessionStorage.setItem(ACCESS_KEY, input.accessToken);
  if (input.refreshToken) {
    sessionStorage.setItem(REFRESH_KEY, input.refreshToken);
  }
  sessionStorage.setItem(USER_KEY, JSON.stringify(input.user));
}

export function clearSession(): void {
  sessionStorage.removeItem(ACCESS_KEY);
  sessionStorage.removeItem(REFRESH_KEY);
  sessionStorage.removeItem(USER_KEY);
  // Conversation UI keys are cleared by ChatProvider on logout.
}
