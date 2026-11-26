import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { streamChat } from '@/services/chat/chatApi';
import { useAuth } from '@/contexts/AuthContext';
import type { ChatMessage, Citation } from '@/types/api';

const SESSION_KEY_PREFIX = 'agentic.web.session.';
const MESSAGES_KEY_PREFIX = 'agentic.web.messages.';

interface ChatContextValue {
  messages: ChatMessage[];
  sessionId: string | null;
  streaming: boolean;
  error: string | null;
  sendMessage: (text: string) => Promise<void>;
  stopGeneration: () => void;
  newChat: () => void;
  clearError: () => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const { user, isAuthenticated } = useAuth();
  const userId = user?.id ?? null;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  /** Guards rapid double-submit of the same logical send. */
  const inFlightRef = useRef(false);

  // Load per-user ephemeral UI state (not server conversation history).
  useEffect(() => {
    if (!userId) {
      setMessages([]);
      setSessionId(null);
      return;
    }
    try {
      const sid = sessionStorage.getItem(SESSION_KEY_PREFIX + userId);
      const raw = sessionStorage.getItem(MESSAGES_KEY_PREFIX + userId);
      setSessionId(sid);
      setMessages(raw ? (JSON.parse(raw) as ChatMessage[]) : []);
    } catch {
      setMessages([]);
      setSessionId(null);
    }
  }, [userId]);

  useEffect(() => {
    if (!userId) return;
    if (sessionId) sessionStorage.setItem(SESSION_KEY_PREFIX + userId, sessionId);
    else sessionStorage.removeItem(SESSION_KEY_PREFIX + userId);
    sessionStorage.setItem(
      MESSAGES_KEY_PREFIX + userId,
      JSON.stringify(messages.filter((m) => !m.streaming))
    );
  }, [userId, sessionId, messages]);

  useEffect(() => {
    if (!isAuthenticated) {
      abortRef.current?.abort();
      setMessages([]);
      setSessionId(null);
      setError(null);
      setStreaming(false);
    }
  }, [isAuthenticated]);

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(null);
    setError(null);
    setStreaming(false);
    if (userId) {
      sessionStorage.removeItem(SESSION_KEY_PREFIX + userId);
      sessionStorage.removeItem(MESSAGES_KEY_PREFIX + userId);
    }
  }, [userId]);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) =>
        m.streaming ? { ...m, streaming: false, content: m.content || '…' } : m
      )
    );
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      // Batch A: one logical send ⇒ one stream POST. No parallel/duplicate submits.
      if (!trimmed || streaming || inFlightRef.current) return;

      setError(null);
      inFlightRef.current = true;

      const history = messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role, content: m.content }));

      const userMsg: ChatMessage = {
        id: uid(),
        role: 'user',
        content: trimmed,
        createdAt: new Date().toISOString(),
      };
      const assistantId = uid();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let reply = '';
      let citations: Citation[] = [];
      let meta: Partial<ChatMessage> = {};
      let rafHandle: number | null = null;
      let pendingReply: string | null = null;
      let sawDone = false;

      const flushTokenUi = () => {
        rafHandle = null;
        if (pendingReply === null) return;
        const content = pendingReply;
        pendingReply = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content, streaming: true }
              : m
          )
        );
      };

      const scheduleTokenUi = (content: string) => {
        pendingReply = content;
        if (rafHandle != null) return;
        rafHandle = window.requestAnimationFrame(flushTokenUi);
      };

      const finalize = (content: string, extra: Partial<ChatMessage> = {}) => {
        if (rafHandle != null) {
          window.cancelAnimationFrame(rafHandle);
          rafHandle = null;
        }
        pendingReply = null;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content,
                  streaming: false,
                  citations: extra.citations ?? citations,
                  contextUsed: extra.contextUsed ?? meta.contextUsed,
                  ragSkipped: extra.ragSkipped ?? meta.ragSkipped,
                  model: extra.model ?? meta.model,
                  provider: extra.provider ?? meta.provider,
                  error: extra.error,
                }
              : m
          )
        );
      };

      try {
        await streamChat(
          {
            message: trimmed,
            history,
            sessionId,
            useRag: true,
            signal: controller.signal,
          },
          (event) => {
            if (event.type === 'heartbeat') {
              return;
            }
            if (event.type === 'meta') {
              if (event.sessionId) setSessionId(event.sessionId);
              citations = event.citations ?? citations;
              meta = {
                contextUsed: event.contextUsed,
                ragSkipped: event.ragSkipped,
                model: event.model,
                provider: event.provider,
              };
            } else if (event.type === 'token') {
              reply += event.token;
              scheduleTokenUi(reply);
            } else if (event.type === 'done') {
              sawDone = true;
              if (event.sessionId) setSessionId(event.sessionId);
              if (event.citations?.length) citations = event.citations;
              meta = {
                ...meta,
                model: event.model ?? meta.model,
                provider: event.provider ?? meta.provider,
              };
            } else if (event.type === 'error') {
              if (event.code === 'CANCELLED') {
                controller.abort();
                return;
              }
              throw new Error(event.error);
            }
          }
        );

        // Empty visible reply is a valid completion — never JSON-retry.
        if (!controller.signal.aborted) {
          finalize(
            reply ||
              (sawDone
                ? ''
                : 'The response stream ended without content. The request was not retried.')
          );
        } else {
          finalize(reply || 'Generation stopped.');
        }
      } catch (err) {
        if (controller.signal.aborted) {
          finalize(reply || 'Generation stopped.');
        } else {
          const message =
            err instanceof Error ? err.message : 'Chat stream interrupted';
          // Never auto-POST /api/chat — stream may already have executed.
          setError(message);
          finalize(
            reply
              ? `${reply}\n\n[Stream interrupted] ${message}`
              : message,
            { error: message }
          );
        }
      } finally {
        if (rafHandle != null) {
          window.cancelAnimationFrame(rafHandle);
          rafHandle = null;
        }
        setStreaming(false);
        abortRef.current = null;
        inFlightRef.current = false;
      }
    },
    [messages, sessionId, streaming]
  );

  const value = useMemo(
    () => ({
      messages,
      sessionId,
      streaming,
      error,
      sendMessage,
      stopGeneration,
      newChat,
      clearError: () => setError(null),
    }),
    [messages, sessionId, streaming, error, sendMessage, stopGeneration, newChat]
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}
