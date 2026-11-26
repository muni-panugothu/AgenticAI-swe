/**
 * Batch A — streamChat performs one POST; sendChat remains a separate intentional path.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const getAccessToken = vi.fn(() => 'tok');
const refreshSession = vi.fn();

vi.mock('@/services/auth/tokenStore', () => ({
  getAccessToken: () => getAccessToken(),
}));

vi.mock('@/services/auth/authApi', () => ({
  refreshSession: (...a: unknown[]) => refreshSession(...a),
}));

vi.mock('@/services/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/services/api/client')>(
    '@/services/api/client'
  );
  return {
    ...actual,
    buildApiUrl: (path: string) => `http://test.local${path}`,
    apiFetch: vi.fn(),
  };
});

describe('chatApi Batch A', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    getAccessToken.mockReturnValue('tok');
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('streamChat posts once and does not call /api/chat', async () => {
    const encoder = new TextEncoder();
    const chunks = [
      'data: {"type":"meta","sessionId":"s1"}\n\n',
      'data: {"type":"token","token":"A"}\n\n',
      'data: {"type":"done","sessionId":"s1"}\n\n',
    ];
    let i = 0;
    const stream = new ReadableStream({
      pull(controller) {
        if (i < chunks.length) {
          controller.enqueue(encoder.encode(chunks[i++]));
        } else {
          controller.close();
        }
      },
    });
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      body: stream,
      json: async () => ({}),
    });

    const { streamChat } = await import('@/services/chat/chatApi');
    const events: string[] = [];
    await streamChat({ message: 'hi' }, (e) => events.push(e.type));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain('/api/chat/stream');
    expect(url).not.toContain('/api/chat"');
    expect(events).toEqual(['meta', 'token', 'done']);
  });

  it('streamChat with empty body still only one POST', async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"type":"meta","sessionId":"s"}\n\ndata: {"type":"done","sessionId":"s"}\n\n'
          )
        );
        controller.close();
      },
    });
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      body: stream,
      json: async () => ({}),
    });

    const { streamChat } = await import('@/services/chat/chatApi');
    await streamChat({ message: 'empty' }, () => undefined);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('intentional sendChat uses /api/chat JSON endpoint', async () => {
    const { apiFetch } = await import('@/services/api/client');
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({
        reply: 'from-json',
        sessionId: 's',
        action: 'general',
        contextUsed: false,
        model: 'm',
      }),
    } as Response);

    const { sendChat } = await import('@/services/chat/chatApi');
    const result = await sendChat({ message: 'json only' });
    expect(result.reply).toBe('from-json');
    expect(apiFetch).toHaveBeenCalledWith(
      '/api/chat',
      expect.objectContaining({ method: 'POST' })
    );
  });
});
