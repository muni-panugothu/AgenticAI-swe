import { apiFetch, apiJson } from '@/services/api/client';
import { getAccessToken } from '@/services/auth/tokenStore';
import { refreshSession } from '@/services/auth/authApi';
import { buildApiUrl } from '@/services/api/client';
import type {
  ChatHistoryItem,
  ChatResponse,
  StreamEvent,
} from '@/types/api';
import { formatApiError } from '@/utils/errors';
import { normalizeChatResponse } from '@/utils/normalize';
import { consumeSseBuffer } from '@/utils/streamParser';

export interface SendChatInput {
  message: string;
  history?: ChatHistoryItem[];
  sessionId?: string | null;
  useRag?: boolean;
  signal?: AbortSignal;
}

export async function sendChat(input: SendChatInput): Promise<ChatResponse> {
  const res = await apiFetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: input.message,
      history: input.history ?? [],
      sessionId: input.sessionId || null,
      useRag: input.useRag ?? true,
    }),
    signal: input.signal,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(formatApiError(res.status, body, res.statusText));
  }
  const raw = (await res.json()) as Record<string, unknown>;
  return normalizeChatResponse(raw);
}

export async function streamChat(
  input: SendChatInput,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };

  let token = getAccessToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let res = await fetch(buildApiUrl('/api/chat/stream'), {
    method: 'POST',
    headers,
    credentials: 'include',
    body: JSON.stringify({
      message: input.message,
      history: input.history ?? [],
      sessionId: input.sessionId || null,
      useRag: input.useRag ?? true,
    }),
    signal: input.signal,
  });

  if (res.status === 401) {
    token = await refreshSession();
    if (!token) {
      const body = await res.json().catch(() => ({}));
      throw new Error(formatApiError(401, body, res.statusText));
    }
    headers.Authorization = `Bearer ${token}`;
    res = await fetch(buildApiUrl('/api/chat/stream'), {
      method: 'POST',
      headers,
      credentials: 'include',
      body: JSON.stringify({
        message: input.message,
        history: input.history ?? [],
        sessionId: input.sessionId || null,
        useRag: input.useRag ?? true,
      }),
      signal: input.signal,
    });
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(formatApiError(res.status, body, res.statusText));
  }

  if (!res.body?.getReader) {
    // Browser without ReadableStream — still parse SSE text progressively
    // via XHR when possible; otherwise one-shot parse (not claimed as live).
    throw new Error(
      'Streaming body unavailable in this browser. Enable ReadableStream support.'
    );
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      if (input.signal?.aborted) {
        try {
          await reader.cancel('aborted');
        } catch {
          // ignore
        }
        onEvent({
          type: 'error',
          error: 'The request was cancelled.',
          code: 'CANCELLED',
        });
        return;
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = consumeSseBuffer(buffer, onEvent);
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      consumeSseBuffer(`${buffer}\n\n`, onEvent);
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

export async function pingChatAuth(): Promise<boolean> {
  try {
    await apiJson('/auth/me', { method: 'GET' });
    return true;
  } catch {
    return false;
  }
}
