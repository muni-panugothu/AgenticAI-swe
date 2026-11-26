import { describe, expect, it, vi } from 'vitest';
import { consumeSseBuffer, parseStreamDataLine } from '@/utils/streamParser';

describe('parseStreamDataLine', () => {
  it('parses meta, token, and done events', () => {
    expect(
      parseStreamDataLine(
        JSON.stringify({
          type: 'meta',
          sessionId: 'abc',
          citations: [],
        })
      )
    ).toMatchObject({ type: 'meta', sessionId: 'abc' });

    expect(
      parseStreamDataLine(JSON.stringify({ type: 'token', token: 'Hello' }))
    ).toEqual({ type: 'token', token: 'Hello' });

    expect(
      parseStreamDataLine(
        JSON.stringify({ type: 'done', sessionId: 'abc', model: 'x' })
      )
    ).toMatchObject({ type: 'done', sessionId: 'abc' });
  });

  it('parses heartbeat and CANCELLED error', () => {
    expect(
      parseStreamDataLine(JSON.stringify({ type: 'heartbeat', ts: 1 }))
    ).toEqual({ type: 'heartbeat', ts: 1 });
    expect(
      parseStreamDataLine(
        JSON.stringify({
          type: 'error',
          code: 'CANCELLED',
          error: 'The request was cancelled.',
        })
      )
    ).toMatchObject({ type: 'error', code: 'CANCELLED' });
  });

  it('returns null for invalid JSON', () => {
    expect(parseStreamDataLine('not-json')).toBeNull();
  });

  it('ignores unknown additive event types', () => {
    expect(
      parseStreamDataLine(
        JSON.stringify({ type: 'structured', responseType: 'answer' })
      )
    ).toBeNull();
  });
});

describe('consumeSseBuffer', () => {
  it('emits complete SSE events and keeps incomplete tail', () => {
    const events: unknown[] = [];
    const leftover = consumeSseBuffer(
      'data: {"type":"token","token":"Hi"}\n\ndata: {"type":"tok',
      (event) => events.push(event)
    );
    expect(events).toEqual([{ type: 'token', token: 'Hi' }]);
    expect(leftover).toContain('data: {"type":"tok');
  });

  it('handles interrupted multi-chunk assembly', () => {
    const onEvent = vi.fn();
    let buf = consumeSseBuffer('data: {"type":"token","token":"A"}\n\n', onEvent);
    buf += 'data: {"type":"token","token":"B"}\n\n';
    buf = consumeSseBuffer(buf, onEvent);
    expect(onEvent).toHaveBeenCalledTimes(2);
    expect(buf).toBe('');
  });

  it('handles multiple events in one chunk', () => {
    const events: unknown[] = [];
    consumeSseBuffer(
      'event: token\ndata: {"type":"token","token":"A"}\n\nevent: token\ndata: {"type":"token","token":"B"}\n\n',
      (e) => events.push(e)
    );
    expect(events).toEqual([
      { type: 'token', token: 'A' },
      { type: 'token', token: 'B' },
    ]);
  });

  it('joins multi-line data fields into one JSON payload', () => {
    const events: unknown[] = [];
    // Spec: multiple data: lines in one event are joined with \n.
    consumeSseBuffer(
      'data: {"type":"token","token":"line1\\nline2"}\n\n',
      (e) => events.push(e)
    );
    expect(events[0]).toEqual({ type: 'token', token: 'line1\nline2' });
  });

  it('ignores heartbeats and comment frames', () => {
    const onEvent = vi.fn();
    consumeSseBuffer(
      ': keep-alive\n\nevent: heartbeat\ndata: {"type":"heartbeat","ts":1}\n\n',
      onEvent
    );
    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent.mock.calls[0][0]).toMatchObject({ type: 'heartbeat' });
  });

  it('maps named cancelled event without JSON type', () => {
    const onEvent = vi.fn();
    consumeSseBuffer('event: cancelled\ndata: {}\n\n', onEvent);
    expect(onEvent).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'error', code: 'CANCELLED' })
    );
  });

  it('handles unicode split across chunks', () => {
    const onEvent = vi.fn();
    // UTF-8 for "é" can split; we operate on already-decoded JS strings here.
    let buf = consumeSseBuffer('data: {"type":"token","token":"caf', onEvent);
    buf += 'é"}\n\n';
    buf = consumeSseBuffer(buf, onEvent);
    expect(onEvent).toHaveBeenCalledWith({ type: 'token', token: 'café' });
    expect(buf).toBe('');
  });

  it('handles stream ending without final newline via caller flush', () => {
    const onEvent = vi.fn();
    let buf = consumeSseBuffer(
      'data: {"type":"token","token":"End"}',
      onEvent
    );
    expect(onEvent).not.toHaveBeenCalled();
    consumeSseBuffer(`${buf}\n\n`, onEvent);
    expect(onEvent).toHaveBeenCalledWith({ type: 'token', token: 'End' });
  });
});
