/**
 * Incremental SSE frame parser for Topic 42 chat streams.
 * One network chunk ≠ one event; frames may split or batch.
 */

export type SseStreamEvent =
  | {
      type: 'meta';
      sessionId?: string;
      action?: string;
      contextUsed?: boolean;
      ragSkipped?: boolean;
      model?: string;
      provider?: string;
      routingReason?: string;
      inputTokensEstimate?: number;
      citations?: unknown;
    }
  | { type: 'token'; token: string }
  | {
      type: 'done';
      sessionId?: string;
      usage?: unknown;
      estimatedCostUsd?: number;
      model?: string;
      provider?: string;
      citations?: unknown;
    }
  | { type: 'error'; error: string; code?: string }
  | { type: 'heartbeat'; ts?: number }
  | { type: 'cancelled'; error?: string; code?: string };

export function parseSseDataPayload(data: string): SseStreamEvent | null {
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
      routingReason:
        typeof raw.routingReason === 'string' ? raw.routingReason : undefined,
      inputTokensEstimate:
        typeof raw.inputTokensEstimate === 'number'
          ? raw.inputTokensEstimate
          : undefined,
      citations: raw.citations,
    };
  }
  if (type === 'done') {
    return {
      type: 'done',
      sessionId: typeof raw.sessionId === 'string' ? raw.sessionId : undefined,
      usage: raw.usage,
      estimatedCostUsd:
        typeof raw.estimatedCostUsd === 'number'
          ? raw.estimatedCostUsd
          : undefined,
      model: typeof raw.model === 'string' ? raw.model : undefined,
      provider: typeof raw.provider === 'string' ? raw.provider : undefined,
      citations: raw.citations,
    };
  }
  if (type === 'heartbeat') {
    return {
      type: 'heartbeat',
      ts: typeof raw.ts === 'number' ? raw.ts : undefined,
    };
  }
  if (type === 'error') {
    const code = typeof raw.code === 'string' ? raw.code : undefined;
    if (code === 'CANCELLED') {
      return {
        type: 'cancelled',
        code,
        error: String(raw.error ?? 'The request was cancelled.'),
      };
    }
    return {
      type: 'error',
      error: String(raw.error ?? 'Stream error'),
      code,
    };
  }
  return null;
}

/**
 * Consume buffered SSE text. Returns leftover incomplete frame text.
 * Supports multi-line `data:` fields joined with `\n`.
 */
export function consumeSseBuffer(
  buffer: string,
  onEvent: (event: SseStreamEvent) => void
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
    const payload = dataLines.join('\n');
    const event = parseSseDataPayload(payload);
    if (!event) {
      // Named CANCELLED without JSON type still maps via event field.
      if (eventName.toLowerCase() === 'cancelled') {
        onEvent({
          type: 'cancelled',
          code: 'CANCELLED',
          error: 'The request was cancelled.',
        });
      }
      continue;
    }
    onEvent(event);
  }

  return incomplete;
}
