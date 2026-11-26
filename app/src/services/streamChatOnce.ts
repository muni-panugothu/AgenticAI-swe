/**
 * Single-POST streaming orchestration for Expo chat (Batch A).
 *
 * Once a POST to `/api/chat/stream` may have been accepted, never submit the
 * same logical send again via XHR or JSON. Empty token buffers never authorize
 * retries — tool-heavy turns often emit no visible tokens until late.
 */

import { consumeSseBuffer, type SseStreamEvent } from './sseParser';

export type StreamTransportKind = 'expo-fetch' | 'xhr';

export class StreamCancelledError extends Error {
  readonly code = 'CANCELLED';
  constructor(message = 'The request was cancelled.') {
    super(message);
    this.name = 'StreamCancelledError';
  }
}

/** No safe streaming transport available *before* any chat POST. */
export class StreamTransportUnavailableError extends Error {
  readonly code = 'STREAM_TRANSPORT_UNAVAILABLE';
  constructor(
    message = 'No progressive streaming transport is available on this device.'
  ) {
    super(message);
    this.name = 'StreamTransportUnavailableError';
  }
}

/**
 * Stream POST was accepted (or may have been), but delivery failed.
 * Must never trigger another chat POST for the same send.
 */
export class StreamDeliveryError extends Error {
  readonly code = 'STREAM_DELIVERY_FAILED';
  readonly partialText: string;
  readonly requestSubmitted: boolean;

  constructor(
    message: string,
    opts?: { partialText?: string; requestSubmitted?: boolean }
  ) {
    super(message);
    this.name = 'StreamDeliveryError';
    this.partialText = opts?.partialText ?? '';
    this.requestSubmitted = opts?.requestSubmitted ?? true;
  }
}

export type StreamTransportCapability = {
  hasReadableStream: boolean;
  hasXmlHttpRequest: boolean;
  expoFetch: typeof fetch | null;
};

export type StreamOnceDeps = {
  detectCapabilities?: () => Promise<StreamTransportCapability>;
  forceTransport?: StreamTransportKind;
  log?: (message: string, detail?: Record<string, string>) => void;
};

export async function detectStreamTransportCapabilities(): Promise<StreamTransportCapability> {
  let expoFetch: typeof fetch | null = null;
  try {
    const mod = await import('expo/fetch');
    if (typeof mod.fetch === 'function') {
      expoFetch = mod.fetch as typeof fetch;
    }
  } catch {
    expoFetch = null;
  }
  return {
    hasReadableStream: typeof ReadableStream !== 'undefined',
    hasXmlHttpRequest: typeof XMLHttpRequest !== 'undefined',
    expoFetch,
  };
}

/** Choose transport *before* any chat POST. */
export function selectStreamTransport(
  caps: StreamTransportCapability
): StreamTransportKind {
  if (caps.expoFetch && caps.hasReadableStream) {
    return 'expo-fetch';
  }
  if (caps.hasXmlHttpRequest) {
    return 'xhr';
  }
  throw new StreamTransportUnavailableError();
}

export type StreamOnceHandlers = {
  signal?: AbortSignal;
  onToken: (token: string, fullText: string) => void;
  onMeta?: (evt: Extract<SseStreamEvent, { type: 'meta' }>) => void;
  onDone?: (evt: Extract<SseStreamEvent, { type: 'done' }>) => void;
  onTransport?: (transport: StreamTransportKind) => void;
};

export type StreamOnceResult = {
  fullText: string;
  meta?: Extract<SseStreamEvent, { type: 'meta' }>;
  done?: Extract<SseStreamEvent, { type: 'done' }>;
  transport: StreamTransportKind;
  requestSubmitted: boolean;
};

type Accumulator = {
  onEvent: (evt: SseStreamEvent) => void;
  getFullText: () => string;
  getMeta: () => Extract<SseStreamEvent, { type: 'meta' }> | undefined;
  getDone: () => Extract<SseStreamEvent, { type: 'done' }> | undefined;
  getTerminalError: () => Error | null;
};

function createAccumulator(handlers: StreamOnceHandlers): Accumulator {
  let fullText = '';
  let meta: Extract<SseStreamEvent, { type: 'meta' }> | undefined;
  let done: Extract<SseStreamEvent, { type: 'done' }> | undefined;
  let terminalError: Error | null = null;

  const onEvent = (evt: SseStreamEvent) => {
    if (evt.type === 'heartbeat') return;
    if (evt.type === 'meta') {
      meta = evt;
      handlers.onMeta?.(evt);
      return;
    }
    if (evt.type === 'token') {
      fullText += evt.token;
      handlers.onToken(evt.token, fullText);
      return;
    }
    if (evt.type === 'done') {
      done = evt;
      handlers.onDone?.(evt);
      return;
    }
    if (evt.type === 'cancelled') {
      terminalError = new StreamCancelledError(evt.error);
      return;
    }
    if (evt.type === 'error') {
      if (evt.code === 'CANCELLED') {
        terminalError = new StreamCancelledError(evt.error);
      } else {
        terminalError = new StreamDeliveryError(evt.error || 'Stream error', {
          partialText: fullText,
          requestSubmitted: true,
        });
      }
    }
  };

  return {
    onEvent,
    getFullText: () => fullText,
    getMeta: () => meta,
    getDone: () => done,
    getTerminalError: () => terminalError,
  };
}

async function abortResponseBody(res: Response): Promise<void> {
  try {
    const reader = res.body?.getReader?.();
    if (reader) {
      await reader.cancel('stream-unreadable');
      return;
    }
    await res.text().catch(() => undefined);
  } catch {
    // ignore
  }
}

export type StreamTransportExecutors = {
  runExpoFetch: (args: {
    url: string;
    body: string;
    headers: Record<string, string>;
    signal?: AbortSignal;
    expoFetch: typeof fetch;
    onEvent: (evt: SseStreamEvent) => void;
    refreshAuth?: () => Promise<Record<string, string> | null>;
  }) => Promise<void>;
  runXhr: (args: {
    url: string;
    body: string;
    headers: Record<string, string>;
    signal?: AbortSignal;
    onEvent: (evt: SseStreamEvent) => void;
    refreshAuth?: () => Promise<Record<string, string> | null>;
  }) => Promise<void>;
};

/** Default expo/fetch runner — one POST; requires getReader after 2xx. */
export async function runExpoFetchStream(args: {
  url: string;
  body: string;
  headers: Record<string, string>;
  signal?: AbortSignal;
  expoFetch: typeof fetch;
  onEvent: (evt: SseStreamEvent) => void;
  refreshAuth?: () => Promise<Record<string, string> | null>;
}): Promise<void> {
  const doFetch = (auth: Record<string, string>) =>
    args.expoFetch(args.url, {
      method: 'POST',
      headers: {
        ...auth,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: args.body,
      signal: args.signal,
    });

  let res = await doFetch(args.headers);
  if (res.status === 401 && args.refreshAuth) {
    const next = await args.refreshAuth();
    if (!next) {
      throw new StreamDeliveryError('Session expired — please sign in again', {
        requestSubmitted: false,
      });
    }
    // 401 ⇒ auth rejected before agent; safe to POST once more on same transport.
    res = await doFetch(next);
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as {
        error?: string;
        detail?: { error?: string };
      };
      detail = String(body.error || body.detail?.error || detail);
    } catch {
      // ignore
    }
    throw new StreamDeliveryError(
      `Stream request failed (HTTP ${res.status}): ${detail}`,
      { requestSubmitted: true }
    );
  }

  const reader = res.body?.getReader?.();
  if (!reader) {
    await abortResponseBody(res);
    throw new StreamDeliveryError(
      'Streaming response was accepted but this runtime cannot read the body progressively. ' +
        'The request was not retried to avoid duplicate agent execution.',
      { requestSubmitted: true }
    );
  }

  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      if (args.signal?.aborted) {
        try {
          await reader.cancel('aborted');
        } catch {
          // ignore
        }
        throw new StreamCancelledError();
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = consumeSseBuffer(buffer, args.onEvent);
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      consumeSseBuffer(`${buffer}\n\n`, args.onEvent);
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

/** Default XHR runner — one POST with onprogress SSE parsing. */
export function runXhrStream(args: {
  url: string;
  body: string;
  headers: Record<string, string>;
  signal?: AbortSignal;
  onEvent: (evt: SseStreamEvent) => void;
  refreshAuth?: () => Promise<Record<string, string> | null>;
}): Promise<void> {
  if (typeof XMLHttpRequest === 'undefined') {
    return Promise.reject(new StreamTransportUnavailableError());
  }

  const sendOnce = (auth: Record<string, string>) =>
    new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let lastIndex = 0;
      let buffer = '';
      let settled = false;

      const finish = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      const fail = (err: Error) => {
        if (settled) return;
        settled = true;
        reject(err);
      };

      const onAbort = () => {
        try {
          xhr.abort();
        } catch {
          // ignore
        }
        fail(new StreamCancelledError());
      };

      if (args.signal) {
        if (args.signal.aborted) {
          fail(new StreamCancelledError());
          return;
        }
        args.signal.addEventListener('abort', onAbort, { once: true });
      }

      xhr.open('POST', args.url, true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.setRequestHeader('Accept', 'text/event-stream');
      for (const [key, value] of Object.entries(auth)) {
        if (key.toLowerCase() === 'content-type') continue;
        if (key.toLowerCase() === 'accept') continue;
        try {
          xhr.setRequestHeader(key, value);
        } catch {
          // ignore
        }
      }
      xhr.responseType = 'text';

      const ingest = (chunk: string) => {
        if (!chunk) return;
        buffer += chunk;
        buffer = consumeSseBuffer(buffer, args.onEvent);
      };

      xhr.onprogress = () => {
        const text = xhr.responseText || '';
        if (text.length <= lastIndex) return;
        ingest(text.slice(lastIndex));
        lastIndex = text.length;
      };

      xhr.onerror = () => {
        fail(
          new StreamDeliveryError('Network request failed during streaming', {
            requestSubmitted: true,
          })
        );
      };

      xhr.onload = () => {
        const text = xhr.responseText || '';
        if (text.length > lastIndex) {
          ingest(text.slice(lastIndex));
          lastIndex = text.length;
        }
        if (buffer.trim()) {
          consumeSseBuffer(`${buffer}\n\n`, args.onEvent);
          buffer = '';
        }
        if (xhr.status === 401) {
          fail(Object.assign(new Error('HTTP 401'), { code: 'HTTP_401' }));
          return;
        }
        if (xhr.status < 200 || xhr.status >= 300) {
          fail(
            new StreamDeliveryError(
              `Stream request failed (HTTP ${xhr.status})`,
              { requestSubmitted: true }
            )
          );
          return;
        }
        finish();
      };

      try {
        xhr.send(args.body);
      } catch (err) {
        fail(
          err instanceof Error
            ? err
            : new StreamDeliveryError(String(err), { requestSubmitted: false })
        );
      }
    });

  return (async () => {
    try {
      await sendOnce(args.headers);
    } catch (err) {
      const is401 =
        err instanceof Error &&
        (err.message === 'HTTP 401' ||
          (err as { code?: string }).code === 'HTTP_401');
      if (is401 && args.refreshAuth) {
        const next = await args.refreshAuth();
        if (!next) {
          throw new StreamDeliveryError(
            'Session expired — please sign in again',
            { requestSubmitted: false }
          );
        }
        await sendOnce(next);
        return;
      }
      throw err;
    }
  })();
}

const defaultExecutors: StreamTransportExecutors = {
  runExpoFetch: runExpoFetchStream,
  runXhr: runXhrStream,
};

/**
 * Select transport, then POST exactly once (auth 401 retry on same transport only).
 * Never falls back to JSON `/api/chat` or a second streaming transport.
 */
export async function streamChatOnce(args: {
  url: string;
  body: string;
  headers: Record<string, string>;
  handlers: StreamOnceHandlers;
  deps?: StreamOnceDeps;
  refreshAuth?: () => Promise<Record<string, string> | null>;
  executors?: Partial<StreamTransportExecutors>;
}): Promise<StreamOnceResult> {
  const caps = await (args.deps?.detectCapabilities
    ? args.deps.detectCapabilities()
    : detectStreamTransportCapabilities());
  const transport =
    args.deps?.forceTransport ?? selectStreamTransport(caps);

  const acc = createAccumulator(args.handlers);
  args.handlers.onTransport?.(transport);
  args.deps?.log?.('chat_stream_transport_selected', { transport });

  const executors: StreamTransportExecutors = {
    ...defaultExecutors,
    ...args.executors,
  };

  try {
    if (transport === 'expo-fetch') {
      if (!caps.expoFetch) {
        throw new StreamTransportUnavailableError(
          'expo/fetch is not available'
        );
      }
      await executors.runExpoFetch({
        url: args.url,
        body: args.body,
        headers: args.headers,
        signal: args.handlers.signal,
        expoFetch: caps.expoFetch,
        onEvent: acc.onEvent,
        refreshAuth: args.refreshAuth,
      });
    } else {
      await executors.runXhr({
        url: args.url,
        body: args.body,
        headers: args.headers,
        signal: args.handlers.signal,
        onEvent: acc.onEvent,
        refreshAuth: args.refreshAuth,
      });
    }
  } catch (err) {
    if (err instanceof StreamCancelledError) throw err;
    if (err instanceof StreamTransportUnavailableError) throw err;
    if (err instanceof StreamDeliveryError) {
      throw new StreamDeliveryError(err.message, {
        partialText: acc.getFullText() || err.partialText,
        requestSubmitted: err.requestSubmitted,
      });
    }
    throw new StreamDeliveryError(
      err instanceof Error ? err.message : 'Stream interrupted',
      { partialText: acc.getFullText(), requestSubmitted: true }
    );
  }

  const terminal = acc.getTerminalError();
  if (terminal) throw terminal;

  return {
    fullText: acc.getFullText(),
    meta: acc.getMeta(),
    done: acc.getDone(),
    transport,
    requestSubmitted: true,
  };
}
