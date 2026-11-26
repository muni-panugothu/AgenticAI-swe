import type { StreamEvent } from '@/types/api';
import { normalizeCitations } from '@/utils/normalize';

/** Parse one SSE `data:` JSON payload into a typed stream event. */
export function parseStreamDataLine(data: string): StreamEvent | null {
  const trimmed = data.trim();
  if (!trimmed || trimmed === '[DONE]') return null;

  let raw: Record<string, unknown>;
  try {
    raw = JSON.parse(trimmed) as Record<string, unknown>;
  } catch {
    return null;
  }

  const type = String(raw.type ?? '');
  if (type === 'token') {
    return { type: 'token', token: String(raw.token ?? '') };
  }
  if (type === 'meta') {
    return {
      type: 'meta',
      sessionId: typeof raw.sessionId === 'string' ? raw.sessionId : undefined,
      action: typeof raw.action === 'string' ? raw.action : undefined,
      contextUsed:
        raw.contextUsed !== undefined ? Boolean(raw.contextUsed) : undefined,
      ragSkipped:
        raw.ragSkipped !== undefined ? Boolean(raw.ragSkipped) : undefined,
      model: typeof raw.model === 'string' ? raw.model : undefined,
      provider: typeof raw.provider === 'string' ? raw.provider : undefined,
      citations: normalizeCitations(raw.citations),
    };
  }
  if (type === 'done') {
    return {
      type: 'done',
      sessionId: typeof raw.sessionId === 'string' ? raw.sessionId : undefined,
      model: typeof raw.model === 'string' ? raw.model : undefined,
      provider: typeof raw.provider === 'string' ? raw.provider : undefined,
      citations: normalizeCitations(raw.citations),
    };
  }
  if (type === 'heartbeat') {
    // Additive lifecycle ping — not user-visible.
    return { type: 'heartbeat', ts: typeof raw.ts === 'number' ? raw.ts : undefined };
  }
  if (type === 'error') {
    const code = typeof raw.code === 'string' ? raw.code : undefined;
    return {
      type: 'error',
      error: String(raw.error ?? 'Stream error'),
      code,
    };
  }
  // Topic 42a: unknown additive event types are ignored safely.
  return null;
}

/**
 * Consume an SSE text buffer, yielding complete events.
 * Returns leftover incomplete buffer text.
 *
 * Handles:
 * - Multiple events in one chunk
 * - Split events across chunks
 * - Multi-line `data:` fields
 * - Comment / heartbeat frames
 * - Named `event:` lines
 */
export function consumeSseBuffer(
  buffer: string,
  onEvent: (event: StreamEvent) => void
): string {
  const parts = buffer.split('\n\n');
  const incomplete = parts.pop() ?? '';
  for (const block of parts) {
    if (!block.trim()) continue;
    const dataLines: string[] = [];
    let eventName = '';
    for (const rawLine of block.split('\n')) {
      const line = rawLine.replace(/\r$/, '');
      if (!line || line.startsWith(':')) continue;
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
        continue;
      }
      if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^\s/, ''));
      }
    }
    if (!dataLines.length) continue;
    const event = parseStreamDataLine(dataLines.join('\n'));
    if (event) {
      onEvent(event);
      continue;
    }
    if (eventName.toLowerCase() === 'cancelled') {
      onEvent({
        type: 'error',
        error: 'The request was cancelled.',
        code: 'CANCELLED',
      });
    }
  }
  return incomplete;
}
