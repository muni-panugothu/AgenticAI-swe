import {
  MessageSquarePlus,
  Settings,
  LogOut,
  Moon,
  Sun,
  PanelLeftClose,
  PanelLeft,
  User,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { BrandMark } from '@/components/common/BrandMark';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useChat } from '@/contexts/ChatContext';

interface SidebarProps {
  open: boolean;
  collapsed: boolean;
  onCloseMobile: () => void;
  onToggleCollapsed: () => void;
}

export function Sidebar({
  open,
  collapsed,
  onCloseMobile,
  onToggleCollapsed,
}: SidebarProps) {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const { newChat, messages } = useChat();

  const name =
    user?.display_name?.trim() || user?.email?.split('@')[0] || 'User';

  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-black/40 transition md:hidden ${
          open ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
        onClick={onCloseMobile}
      />

      <aside
        className={[
          'fixed inset-y-0 left-0 z-50 flex flex-col border-r border-[var(--border)] bg-[var(--sidebar)] transition-all duration-200 md:static md:translate-x-0',
          collapsed ? 'md:w-[72px]' : 'md:w-[280px]',
          open ? 'translate-x-0 w-[280px]' : '-translate-x-full w-[280px]',
        ].join(' ')}
      >
        <div className="flex items-center gap-3 px-4 py-4">
          <BrandMark size={32} />
          {!collapsed && (
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-[var(--text)]">
                Agentic Engineer
              </p>
              <p className="truncate text-xs text-[var(--text-muted)]">
                AI workspace
              </p>
            </div>
          )}
          <button
            type="button"
            className="hidden rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--surface-muted)] md:inline-flex"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeft size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>

        <div className="px-3">
          <button
            type="button"
            onClick={() => {
              newChat();
              onCloseMobile();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-3 py-2.5 text-sm font-medium text-[var(--bg)] hover:opacity-90"
          >
            <MessageSquarePlus size={18} />
            {!collapsed && <span>New chat</span>}
          </button>
        </div>

        <div className="mt-4 flex-1 overflow-y-auto px-3">
          {!collapsed && (
            <div className="rounded-xl border border-dashed border-[var(--border)] px-3 py-4 text-xs leading-5 text-[var(--text-muted)]">
              <p className="mb-1 font-medium text-[var(--text-secondary)]">
                Conversation history
              </p>
              <p>
                The backend does not expose persistent chat history yet. This
                session keeps {messages.length}{' '}
                {messages.length === 1 ? 'message' : 'messages'} in temporary
                browser storage for your account.
              </p>
            </div>
          )}
        </div>

        <div className="space-y-1 border-t border-[var(--border)] p-3">
          <NavLink
            to="/settings"
            onClick={onCloseMobile}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm ${
                isActive
                  ? 'bg-[var(--surface-muted)] text-[var(--text)]'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--surface-muted)]'
              }`
            }
          >
            <Settings size={18} />
            {!collapsed && <span>Settings</span>}
          </NavLink>

          <button
            type="button"
            onClick={toggleTheme}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[var(--text-secondary)] hover:bg-[var(--surface-muted)]"
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            {!collapsed && (
              <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
            )}
          </button>

          <div
            className={`flex items-center gap-3 rounded-xl px-3 py-2.5 ${
              collapsed ? 'justify-center' : ''
            }`}
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--surface-muted)]">
              <User size={16} />
            </div>
            {!collapsed && (
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-[var(--text)]">
                  {name}
                </p>
                <p className="truncate text-xs text-[var(--text-muted)]">
                  {user?.email}
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => void logout()}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[var(--text-secondary)] hover:bg-[var(--surface-muted)]"
          >
            <LogOut size={18} />
            {!collapsed && <span>Log out</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
