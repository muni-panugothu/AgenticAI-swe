import { useEffect, useState } from 'react';
import { checkHealth, getApiBaseUrl } from '@/services/api/client';
import { DocumentPanel } from '@/components/documents/DocumentPanel';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useChat } from '@/contexts/ChatContext';

export function SettingsPage() {
  const { user } = useAuth();
  const { theme, setTheme } = useTheme();
  const { sessionId, newChat } = useChat();
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    void checkHealth().then((ok) => {
      if (!cancelled) setHealthy(ok);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="h-full overflow-y-auto px-4 py-8">
      <div className="mx-auto w-full max-w-3xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-[var(--text)]">Settings</h1>
          <p className="mt-1 text-sm text-[var(--text-muted)]">
            Profile, theme, and connection details for this workspace.
          </p>
        </div>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
          <h2 className="text-sm font-semibold text-[var(--text)]">Profile</h2>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">Name</dt>
              <dd>{user?.display_name || '—'}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">Email</dt>
              <dd className="truncate">{user?.email}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">User ID</dt>
              <dd className="truncate font-mono text-xs">{user?.id}</dd>
            </div>
          </dl>
        </section>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
          <h2 className="text-sm font-semibold text-[var(--text)]">Appearance</h2>
          <div className="mt-3 flex gap-2">
            {(['light', 'dark'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setTheme(mode)}
                className={`rounded-xl px-4 py-2 text-sm capitalize ${
                  theme === mode
                    ? 'bg-[var(--accent)] text-[var(--bg)]'
                    : 'bg-[var(--surface-muted)] text-[var(--text-secondary)]'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5">
          <h2 className="text-sm font-semibold text-[var(--text)]">API connection</h2>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">Base URL</dt>
              <dd className="truncate font-mono text-xs">{getApiBaseUrl()}</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">Health</dt>
              <dd>
                {healthy === null
                  ? 'Checking…'
                  : healthy
                    ? 'Reachable'
                    : 'Unreachable'}
              </dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-[var(--text-muted)]">Chat session</dt>
              <dd className="truncate font-mono text-xs">
                {sessionId || 'None (new chat)'}
              </dd>
            </div>
          </dl>
          <button
            type="button"
            onClick={newChat}
            className="mt-4 rounded-xl border border-[var(--border)] px-3 py-2 text-sm hover:bg-[var(--surface-muted)]"
          >
            Clear current conversation
          </button>
          <p className="mt-3 text-xs leading-5 text-[var(--text-muted)]">
            Conversation messages are stored only in temporary browser storage
            for the signed-in user. The backend does not provide a persistent
            conversation list API.
          </p>
        </section>

        <DocumentPanel />
      </div>
    </div>
  );
}
