/**
 * Batch A — Web chat must never JSON-fallback after a stream POST may have run.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, waitFor, act } from '@testing-library/react';
import { ChatProvider, useChat } from '@/contexts/ChatContext';
import type { StreamEvent } from '@/types/api';

const streamChat = vi.fn();
const sendChat = vi.fn();

vi.mock('@/services/chat/chatApi', () => ({
  streamChat: (...args: unknown[]) => streamChat(...args),
  sendChat: (...args: unknown[]) => sendChat(...args),
}));

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-batch-a', email: 'a@b.c', displayName: 'A' },
    isAuthenticated: true,
    loading: false,
  }),
}));

function Probe({
  onReady,
}: {
  onReady: (api: ReturnType<typeof useChat>) => void;
}) {
  const api = useChat();
  onReady(api);
  return (
    <div>
      <span data-testid="streaming">{String(api.streaming)}</span>
      <span data-testid="error">{api.error ?? ''}</span>
      <ul>
        {api.messages.map((m) => (
          <li key={m.id} data-role={m.role}>
            {m.content}
          </li>
        ))}
      </ul>
    </div>
  );
}

describe('ChatContext Batch A — one stream POST, no JSON fallback', () => {
  let api: ReturnType<typeof useChat> | null = null;

  beforeEach(() => {
    api = null;
    streamChat.mockReset();
    sendChat.mockReset();
    sessionStorage.clear();
    vi.stubGlobal(
      'requestAnimationFrame',
      (cb: FrameRequestCallback) => {
        cb(0);
        return 1;
      }
    );
    vi.stubGlobal('cancelAnimationFrame', () => undefined);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function mount() {
    render(
      <ChatProvider>
        <Probe
          onReady={(a) => {
            api = a;
          }}
        />
      </ChatProvider>
    );
  }

  it('empty visible reply after stream completion — no JSON fallback', async () => {
    streamChat.mockImplementation(
      async (
        _input: unknown,
        onEvent: (e: StreamEvent) => void
      ) => {
        onEvent({
          type: 'meta',
          sessionId: 's1',
          contextUsed: false,
        });
        onEvent({ type: 'done', sessionId: 's1', model: 'm' });
      }
    );
    mount();
    await act(async () => {
      await api!.sendMessage('tool heavy');
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
  });

  it('stream error after backend may have started — no JSON fallback', async () => {
    streamChat.mockImplementation(async () => {
      throw new Error('network reset after accept');
    });
    mount();
    await act(async () => {
      await api!.sendMessage('hello');
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(api!.error).toMatch(/network reset/i);
    });
  });

  it('partial tokens then error — preserves partial, one POST only', async () => {
    streamChat.mockImplementation(
      async (
        _input: unknown,
        onEvent: (e: StreamEvent) => void
      ) => {
        onEvent({ type: 'token', token: 'Partial' });
        onEvent({ type: 'token', token: ' answer' });
        throw new Error('stream cut');
      }
    );
    mount();
    await act(async () => {
      await api!.sendMessage('ask');
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
    await waitFor(() => {
      const assistant = api!.messages.find((m) => m.role === 'assistant');
      expect(assistant?.content).toContain('Partial answer');
      expect(assistant?.content).toContain('Stream interrupted');
    });
  });

  it('rapid duplicate submission — one stream request', async () => {
    let resolveStream!: () => void;
    const gate = new Promise<void>((r) => {
      resolveStream = r;
    });
    streamChat.mockImplementation(async () => {
      await gate;
    });
    mount();
    await act(async () => {
      const p1 = api!.sendMessage('once');
      const p2 = api!.sendMessage('once');
      resolveStream();
      await Promise.all([p1, p2]);
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
  });

  it('successful streaming — tokens, done, citations, session', async () => {
    streamChat.mockImplementation(
      async (
        _input: unknown,
        onEvent: (e: StreamEvent) => void
      ) => {
        onEvent({
          type: 'meta',
          sessionId: 'sess-9',
          contextUsed: true,
          citations: [
            {
              index: 1,
              documentId: 'd1',
              chunkIndex: 0,
              excerpt: 'ex',
              score: 0.9,
            },
          ],
          model: 'llama',
          provider: 'groq',
        });
        onEvent({ type: 'token', token: 'Hi' });
        onEvent({ type: 'token', token: '!' });
        onEvent({
          type: 'done',
          sessionId: 'sess-9',
          model: 'llama',
          provider: 'groq',
          citations: [
            {
              index: 1,
              documentId: 'd1',
              chunkIndex: 0,
              excerpt: 'ex',
              score: 0.9,
            },
          ],
        });
      }
    );
    mount();
    await act(async () => {
      await api!.sendMessage('hi');
    });
    expect(streamChat).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(api!.sessionId).toBe('sess-9');
      const assistant = api!.messages.find((m) => m.role === 'assistant');
      expect(assistant?.content).toBe('Hi!');
      expect(assistant?.citations?.[0]?.documentId).toBe('d1');
      expect(assistant?.streaming).toBe(false);
    });
  });
});

describe('sendChat JSON path remains available', () => {
  it('sendChat can still be invoked intentionally', async () => {
    const { sendChat: realSend } = await import('@/services/chat/chatApi');
    // Module is mocked — verify the export exists for intentional JSON use.
    expect(typeof realSend).toBe('function');
    sendChat.mockResolvedValue({
      reply: 'json ok',
      sessionId: 's',
      action: 'general',
      contextUsed: false,
      model: 'm',
    });
    const result = await sendChat({ message: 'direct json' });
    expect(result.reply).toBe('json ok');
    expect(sendChat).toHaveBeenCalledTimes(1);
  });
});
