import { buildApiUrl } from './apiClient';
import { getAuthHeaders } from './authStorage';
import { refreshAccessToken } from './authApi';

export async function fetchWithAuth(
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  const url = buildApiUrl(path);
  const baseHeaders = await getAuthHeaders();
  const extra = (init.headers ?? {}) as Record<string, string>;
  let res = await fetch(url, {
    ...init,
    credentials: 'include',
    headers: { ...baseHeaders, ...extra },
  });
  if (res.status === 401) {
    const renewed = await refreshAccessToken();
    if (!renewed) {
      throw new Error('Session expired — please sign in again');
    }
    const retryHeaders = await getAuthHeaders();
    res = await fetch(url, {
      ...init,
      credentials: 'include',
      headers: { ...retryHeaders, ...extra },
    });
  }
  return res;
}
