/**
 * Auth token storage — file-backed (no expo-secure-store import).
 *
 * IMPORTANT: Never `require('expo-secure-store')` unless the native module is
 * already linked. That package calls requireNativeModule('ExpoSecureStore') at
 * load time and crashes red-screen when the installed binary lacks it.
 *
 * After `npx expo run:android` (or ios) rebuilds with SecureStore linked, set
 * EXPO_PUBLIC_USE_SECURE_STORE=1 to prefer hardware-backed storage via a
 * guarded optional native lookup (still no hard dependency on the JS package
 * until the native module exists).
 */

import { NativeModules } from 'react-native';

export interface StoredUser {
  id: string;
  email: string;
  display_name?: string | null;
  tenant_id?: string | null;
  status?: string;
  terms_version?: string | null;
  privacy_version?: string | null;
  terms_accepted_at?: string | null;
  privacy_accepted_at?: string | null;
}

const ACCESS_KEY = 'agentic_access_token';
const REFRESH_KEY = 'agentic_refresh_token';
const USER_KEY = 'agentic_user_json';
const GREETING_KEY = 'agentic_last_greeting_name';
const REMEMBER_EMAIL_KEY = 'agentic_remembered_email';
const FALLBACK_FILE = 'agentic_auth_session.json';
const GREETING_FILE = 'agentic_last_greeting.json';
const REMEMBER_FILE = 'agentic_remembered_email.json';

interface SessionBlob {
  accessToken?: string | null;
  refreshToken?: string | null;
  userJson?: string | null;
}

type SecureLike = {
  getItemAsync: (key: string) => Promise<string | null>;
  setItemAsync: (key: string, value: string) => Promise<void>;
  deleteItemAsync: (key: string) => Promise<void>;
};

let memoryFallback: SessionBlob = {};
let secureImpl: SecureLike | null | undefined;
let warned = false;

function nativeSecureStorePresent(): boolean {
  const modules = NativeModules as Record<string, unknown>;
  return Boolean(modules.ExpoSecureStore);
}

/**
 * Only load the SecureStore JS package when the native module is already in
 * the binary. Otherwise skip entirely (never call require on the package).
 */
function getSecureStore(): SecureLike | null {
  if (secureImpl !== undefined) return secureImpl;

  if (!nativeSecureStorePresent()) {
    secureImpl = null;
    return null;
  }

  // Opt-in so accidental requires never happen on old binaries.
  if (process.env.EXPO_PUBLIC_USE_SECURE_STORE !== '1') {
    secureImpl = null;
    return null;
  }

  try {
    // Native module exists — safe to load the JS wrapper.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require('expo-secure-store') as SecureLike;
    if (typeof mod?.getItemAsync !== 'function') {
      secureImpl = null;
      return null;
    }
    secureImpl = mod;
    return secureImpl;
  } catch {
    secureImpl = null;
    return null;
  }
}

function warnFileBackendOnce(): void {
  if (warned || !__DEV__) return;
  warned = true;
  console.warn(
    '[authStorage] Using private document-file token storage. ' +
      'Rebuild native app (`npx expo run:android`) then set ' +
      'EXPO_PUBLIC_USE_SECURE_STORE=1 for Keystore-backed SecureStore.'
  );
}

async function readBlob(): Promise<SessionBlob> {
  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, FALLBACK_FILE);
    if (!file.exists) return { ...memoryFallback };
    const raw = await file.text();
    const parsed = JSON.parse(raw) as SessionBlob;
    memoryFallback = parsed;
    return parsed;
  } catch {
    return { ...memoryFallback };
  }
}

async function writeBlob(blob: SessionBlob): Promise<void> {
  memoryFallback = blob;
  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, FALLBACK_FILE);
    if (!file.exists) {
      file.create();
    }
    file.write(JSON.stringify(blob));
  } catch (err) {
    if (__DEV__) {
      console.warn('[authStorage] file write failed; using memory only', err);
    }
  }
}

/** First name / local-part for "Welcome Back, {name}!" after logout. */
export function greetingNameFromUser(user: StoredUser): string {
  const display = user.display_name?.trim();
  if (display) {
    return display.split(/\s+/)[0] ?? display;
  }
  const local = user.email?.split('@')[0]?.trim();
  if (local) {
    return local.charAt(0).toUpperCase() + local.slice(1);
  }
  return '';
}

export async function saveLastGreetingName(name: string): Promise<void> {
  const trimmed = name.trim();
  if (!trimmed) return;

  const store = getSecureStore();
  if (store) {
    await store.setItemAsync(GREETING_KEY, trimmed);
    return;
  }

  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, GREETING_FILE);
    if (!file.exists) file.create();
    file.write(JSON.stringify({ name: trimmed }));
  } catch (err) {
    if (__DEV__) {
      console.warn('[authStorage] greeting write failed', err);
    }
  }
}

export async function getLastGreetingName(): Promise<string | null> {
  const store = getSecureStore();
  if (store) {
    const raw = await store.getItemAsync(GREETING_KEY);
    return raw?.trim() || null;
  }

  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, GREETING_FILE);
    if (!file.exists) return null;
    const parsed = JSON.parse(await file.text()) as { name?: string };
    return parsed.name?.trim() || null;
  } catch {
    return null;
  }
}

/** Persist email when Remember Me is checked (never store the password). */
export async function saveRememberedEmail(email: string | null): Promise<void> {
  const store = getSecureStore();
  if (store) {
    if (!email) {
      await store.deleteItemAsync(REMEMBER_EMAIL_KEY);
      return;
    }
    await store.setItemAsync(REMEMBER_EMAIL_KEY, email.trim());
    return;
  }

  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, REMEMBER_FILE);
    if (!email) {
      if (file.exists) file.delete();
      return;
    }
    if (!file.exists) file.create();
    file.write(JSON.stringify({ email: email.trim() }));
  } catch (err) {
    if (__DEV__) {
      console.warn('[authStorage] remember email write failed', err);
    }
  }
}

export async function getRememberedEmail(): Promise<string | null> {
  const store = getSecureStore();
  if (store) {
    const raw = await store.getItemAsync(REMEMBER_EMAIL_KEY);
    return raw?.trim() || null;
  }

  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, REMEMBER_FILE);
    if (!file.exists) return null;
    const parsed = JSON.parse(await file.text()) as { email?: string };
    return parsed.email?.trim() || null;
  } catch {
    return null;
  }
}

export async function saveTokens(session: {
  accessToken: string;
  refreshToken?: string | null;
  user: StoredUser;
}): Promise<void> {
  const greeting = greetingNameFromUser(session.user);
  if (greeting) {
    await saveLastGreetingName(greeting);
  }

  const store = getSecureStore();
  if (store) {
    await store.setItemAsync(ACCESS_KEY, session.accessToken);
    if (session.refreshToken) {
      await store.setItemAsync(REFRESH_KEY, session.refreshToken);
    }
    await store.setItemAsync(USER_KEY, JSON.stringify(session.user));
    return;
  }

  warnFileBackendOnce();
  await writeBlob({
    accessToken: session.accessToken,
    refreshToken: session.refreshToken ?? null,
    userJson: JSON.stringify(session.user),
  });
}

export async function getAccessToken(): Promise<string | null> {
  const store = getSecureStore();
  if (store) return store.getItemAsync(ACCESS_KEY);
  warnFileBackendOnce();
  return (await readBlob()).accessToken ?? null;
}

export async function getRefreshToken(): Promise<string | null> {
  const store = getSecureStore();
  if (store) return store.getItemAsync(REFRESH_KEY);
  warnFileBackendOnce();
  return (await readBlob()).refreshToken ?? null;
}

export async function getStoredUser(): Promise<StoredUser | null> {
  const store = getSecureStore();
  let raw: string | null = null;
  if (store) {
    raw = await store.getItemAsync(USER_KEY);
  } else {
    warnFileBackendOnce();
    raw = (await readBlob()).userJson ?? null;
  }
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredUser;
  } catch {
    return null;
  }
}

export async function clearTokens(): Promise<void> {
  const store = getSecureStore();
  if (store) {
    await store.deleteItemAsync(ACCESS_KEY);
    await store.deleteItemAsync(REFRESH_KEY);
    await store.deleteItemAsync(USER_KEY);
    return;
  }
  warnFileBackendOnce();
  memoryFallback = {};
  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, FALLBACK_FILE);
    if (file.exists) file.delete();
  } catch {
    // ignore
  }
}

export async function getAuthHeaders(): Promise<Record<string, string>> {
  const token = await getAccessToken();
  if (!token) return { Accept: 'application/json' };
  return {
    Accept: 'application/json',
    Authorization: `Bearer ${token}`,
  };
}

export async function isSecureStoreActive(): Promise<boolean> {
  return getSecureStore() != null;
}
