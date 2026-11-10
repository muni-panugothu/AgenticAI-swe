import * as SecureStore from 'expo-secure-store';

const ACCESS_KEY = 'agentic_access_token';
const REFRESH_KEY = 'agentic_refresh_token';
const USER_KEY = 'agentic_user_json';

export interface StoredUser {
  id: string;
  email: string;
  display_name?: string | null;
  tenant_id?: string | null;
  status?: string;
}

export async function saveTokens(session: {
  accessToken: string;
  refreshToken?: string | null;
  user: StoredUser;
}): Promise<void> {
  await SecureStore.setItemAsync(ACCESS_KEY, session.accessToken);
  if (session.refreshToken) {
    await SecureStore.setItemAsync(REFRESH_KEY, session.refreshToken);
  }
  await SecureStore.setItemAsync(USER_KEY, JSON.stringify(session.user));
}

export async function getAccessToken(): Promise<string | null> {
  return SecureStore.getItemAsync(ACCESS_KEY);
}

export async function getRefreshToken(): Promise<string | null> {
  return SecureStore.getItemAsync(REFRESH_KEY);
}

export async function getStoredUser(): Promise<StoredUser | null> {
  const raw = await SecureStore.getItemAsync(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredUser;
  } catch {
    return null;
  }
}

export async function clearTokens(): Promise<void> {
  await SecureStore.deleteItemAsync(ACCESS_KEY);
  await SecureStore.deleteItemAsync(REFRESH_KEY);
  await SecureStore.deleteItemAsync(USER_KEY);
}

export async function getAuthHeaders(): Promise<Record<string, string>> {
  const token = await getAccessToken();
  if (!token) return { Accept: 'application/json' };
  return {
    Accept: 'application/json',
    Authorization: `Bearer ${token}`,
  };
}
