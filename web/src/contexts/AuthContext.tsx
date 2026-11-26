import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import * as authApi from '@/services/auth/authApi';
import { configureAuthHandlers } from '@/services/api/client';
import {
  clearSession,
  getAccessToken,
  getStoredUser,
} from '@/services/auth/tokenStore';
import type { AuthUser } from '@/types/api';

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string, rememberMe?: boolean) => Promise<void>;
  register: (input: {
    displayName: string;
    email: string;
    password: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => getStoredUser());
  const [loading, setLoading] = useState(true);

  const handleAuthFailure = useCallback(() => {
    clearSession();
    setUser(null);
  }, []);

  useEffect(() => {
    configureAuthHandlers({
      getAccessToken,
      refreshAccessToken: authApi.refreshSession,
      onAuthFailure: handleAuthFailure,
    });
  }, [handleAuthFailure]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const token = getAccessToken();
      if (!token) {
        if (!cancelled) {
          setUser(null);
          setLoading(false);
        }
        return;
      }
      try {
        const me = await authApi.fetchMe();
        if (!cancelled) setUser(me);
      } catch {
        const refreshed = await authApi.refreshSession();
        if (!refreshed) {
          if (!cancelled) {
            setUser(null);
          }
        } else {
          try {
            const me = await authApi.fetchMe();
            if (!cancelled) setUser(me);
          } catch {
            if (!cancelled) setUser(null);
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string, rememberMe = false) => {
      const next = await authApi.login({ email, password, rememberMe });
      setUser(next);
    },
    []
  );

  const register = useCallback(
    async (input: { displayName: string; email: string; password: string }) => {
      const next = await authApi.register({
        displayName: input.displayName,
        email: input.email,
        password: input.password,
        acceptedTerms: true,
        acceptedPrivacy: true,
      });
      setUser(next);
    },
    []
  );

  const logout = useCallback(async () => {
    await authApi.logout();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const me = await authApi.fetchMe();
    setUser(me);
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      isAuthenticated: Boolean(user),
      login,
      register,
      logout,
      refreshUser,
    }),
    [user, loading, login, register, logout, refreshUser]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
