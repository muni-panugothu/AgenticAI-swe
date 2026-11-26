import {
  discoverApiBaseUrl,
  buildApiUrl,
  getConnectionHelpMessage,
} from './apiClient';
import {
  formatApiError,
  normalizeCitations,
  normalizeDocumentList,
  normalizeUploadResponse,
} from './apiNormalize';
import { fetchWithAuth } from './authorizedFetch';
import { getAuthHeaders } from './authStorage';
import { refreshAccessToken } from './authApi';
import {
  StreamCancelledError,
  StreamDeliveryError,
  StreamTransportUnavailableError,
  streamChatOnce,
  type StreamTransportKind,
} from './streamChatOnce';

export {
  StreamCancelledError,
  StreamDeliveryError,
  StreamTransportUnavailableError,
};

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface Citation {
  index: number;
  documentId: string;
  source?: string;
  section?: string;
  chunkIndex: number;
  excerpt: string;
  score: number;
}

export interface ChatResponse {
  reply: string;
  sessionId?: string;
  action: string;
  contextUsed: boolean;
  ragSkipped?: boolean;
  citations?: Citation[];
  model: string;
  provider?: string;
  routingReason?: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  estimatedCostUsd?: number;
  inputTokensEstimate?: number;
}

export type StreamEvent =
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
      citations?: Citation[];
    }
  | { type: 'token'; token: string }
  | {
      type: 'done';
      sessionId?: string;
      usage?: ChatResponse['usage'];
      estimatedCostUsd?: number;
      model?: string;
      provider?: string;
      citations?: Citation[];
    }
  | {
      type: 'error';
      error: string;
      code?: string;
    };

export interface UploadResponse {
  documentId: string;
  strategy: string;
  chunkCount: number;
  filename: string;
}

async function ensureConnected(): Promise<void> {
  await discoverApiBaseUrl();
}

function classifyClientError(err: unknown): Error {
  const detail = err instanceof Error ? err.message : String(err);

  if (detail.includes('AI provider')) return new Error(detail);
  if (detail.includes('RAG service')) return new Error(detail);
  if (detail.includes('Database error')) return new Error(detail);
  if (detail.includes('Server error')) return new Error(detail);
  if (detail.includes('authentication')) return new Error(detail);
  if (detail.includes('rate limit')) return new Error(detail);

  if (detail.includes('Network request failed') || detail.includes('Failed to fetch')) {
    return new Error(getConnectionHelpMessage());
  }

  return new Error(`fetch failed: ${detail}`);
}

function normalizeChatResponse(raw: Record<string, unknown>): ChatResponse {
  return {
    reply: String(raw.reply ?? ''),
    sessionId:
      typeof raw.sessionId === 'string'
        ? raw.sessionId
        : typeof raw.session_id === 'string'
          ? raw.session_id
          : undefined,
    action: String(raw.action ?? 'general'),
    contextUsed: Boolean(raw.contextUsed ?? raw.context_used),
    ragSkipped:
      raw.ragSkipped !== undefined || raw.rag_skipped !== undefined
        ? Boolean(raw.ragSkipped ?? raw.rag_skipped)
        : undefined,
    citations: normalizeCitations(raw.citations),
    model: String(raw.model ?? 'unknown'),
    provider:
      typeof raw.provider === 'string' ? raw.provider : undefined,
    routingReason:
      typeof raw.routingReason === 'string'
        ? raw.routingReason
        : typeof raw.routing_reason === 'string'
          ? raw.routing_reason
          : undefined,
    usage:
      raw.usage && typeof raw.usage === 'object'
        ? (raw.usage as ChatResponse['usage'])
        : undefined,
    estimatedCostUsd:
      typeof raw.estimatedCostUsd === 'number'
        ? raw.estimatedCostUsd
        : undefined,
    inputTokensEstimate:
      typeof raw.inputTokensEstimate === 'number'
        ? raw.inputTokensEstimate
        : undefined,
  };
}

async function parseJsonResponse(res: Response): Promise<Record<string, unknown>> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(formatApiError(res.status, body, res.statusText));
  }
  return (await res.json()) as Record<string, unknown>;
}

async function authHeadersForUpload(): Promise<Record<string, string>> {
  let headers = await getAuthHeaders();
  if (!headers.Authorization) {
    const renewed = await refreshAccessToken();
    if (!renewed) {
      throw new Error('Session expired — please sign in again');
    }
    headers = await getAuthHeaders();
  }
  return headers;
}

export async function sendChatMessage(
  message: string,
  history: ChatHistoryItem[] = [],
  options?: {
    action?: string;
    useRag?: boolean;
    sessionId?: string;
    preferProvider?: 'groq' | 'openai';
    preferModel?: string;
  }
): Promise<ChatResponse> {
  await ensureConnected();
  try {
    const res = await fetchWithAuth('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history,
        action: options?.action,
        useRag: options?.useRag ?? true,
        sessionId: options?.sessionId,
        preferProvider: options?.preferProvider,
        preferModel: options?.preferModel,
      }),
    });
    return normalizeChatResponse(await parseJsonResponse(res));
  } catch (err) {
    throw classifyClientError(err);
  }
}

type StreamHandlers = {
  action?: string;
  useRag?: boolean;
  sessionId?: string;
  preferProvider?: 'groq' | 'openai';
  preferModel?: string;
  signal?: AbortSignal;
  onToken: (token: string, fullText: string) => void;
  onMeta?: (meta: Extract<StreamEvent, { type: 'meta' }>) => void;
  onDone?: (done: Extract<StreamEvent, { type: 'done' }>) => void;
  /** Selected streaming transport (never json-fallback). */
  onTransport?: (transport: StreamTransportKind) => void;
};

function buildChatBody(
  message: string,
  history: ChatHistoryItem[],
  options: StreamHandlers
) {
  return JSON.stringify({
    message,
    history,
    action: options.action,
    useRag: options.useRag ?? true,
    sessionId: options.sessionId,
    preferProvider: options.preferProvider,
    preferModel: options.preferModel,
  });
}

async function authHeadersForStream(): Promise<Record<string, string>> {
  let headers = await getAuthHeaders();
  if (!headers.Authorization) {
    const renewed = await refreshAccessToken();
    if (!renewed) {
      throw new Error('Session expired — please sign in again');
    }
    headers = await getAuthHeaders();
  }
  return headers;
}

/**
 * Stream chat tokens via SSE from POST /api/chat/stream.
 *
 * Batch A: select one transport *before* POST; never auto-fallback to XHR or
 * JSON after a streaming request may have been accepted. Empty token buffers
 * do not authorize retries.
 */
export async function streamChatMessage(
  message: string,
  history: ChatHistoryItem[] = [],
  options: StreamHandlers
): Promise<ChatResponse> {
  await ensureConnected();
  const url = buildApiUrl('/api/chat/stream');
  const body = buildChatBody(message, history, options);

  try {
    const headers = await authHeadersForStream();
    const result = await streamChatOnce({
      url,
      body,
      headers,
      handlers: {
        signal: options.signal,
        onToken: options.onToken,
        onMeta: (meta) => {
          options.onMeta?.({
            type: 'meta',
            sessionId: meta.sessionId,
            action: meta.action,
            contextUsed: meta.contextUsed,
            ragSkipped: meta.ragSkipped,
            model: meta.model,
            provider: meta.provider,
            routingReason: meta.routingReason,
            inputTokensEstimate: meta.inputTokensEstimate,
            citations: normalizeCitations(meta.citations),
          });
        },
        onDone: (done) => {
          options.onDone?.({
            type: 'done',
            sessionId: done.sessionId,
            usage: done.usage as ChatResponse['usage'],
            estimatedCostUsd: done.estimatedCostUsd,
            model: done.model,
            provider: done.provider,
            citations: normalizeCitations(done.citations),
          });
        },
        onTransport: options.onTransport,
      },
      refreshAuth: async () => {
        const renewed = await refreshAccessToken();
        if (!renewed) return null;
        return getAuthHeaders();
      },
      deps: {
        log: (msg, detail) => {
          if (__DEV__) {
            console.log(`[chat] ${msg}`, detail ?? '');
          }
        },
      },
    });

    return normalizeChatResponse({
      reply: result.fullText,
      sessionId: result.done?.sessionId ?? result.meta?.sessionId,
      action: result.meta?.action ?? 'general',
      contextUsed: result.meta?.contextUsed ?? false,
      ragSkipped: result.meta?.ragSkipped,
      model: result.done?.model ?? result.meta?.model ?? 'unknown',
      provider: result.done?.provider ?? result.meta?.provider,
      routingReason: result.meta?.routingReason,
      usage: result.done?.usage as ChatResponse['usage'],
      estimatedCostUsd: result.done?.estimatedCostUsd,
      inputTokensEstimate: result.meta?.inputTokensEstimate,
      citations: normalizeCitations(
        result.done?.citations ?? result.meta?.citations
      ),
    });
  } catch (err) {
    if (err instanceof StreamCancelledError) throw err;
    if (err instanceof StreamTransportUnavailableError) {
      throw classifyClientError(err);
    }
    if (err instanceof StreamDeliveryError) {
      // Partial text already pushed via onToken — surface delivery error only.
      throw err;
    }
    throw classifyClientError(err);
  }
}

export async function sendActionMessage(
  actionId: string,
  message?: string,
  history: ChatHistoryItem[] = [],
  sessionId?: string
): Promise<ChatResponse> {
  await ensureConnected();
  try {
    const res = await fetchWithAuth(`/api/chat/action/${actionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message && message.trim() ? message : 'Continue',
        history,
        sessionId,
        action: actionId,
      }),
    });
    return normalizeChatResponse(await parseJsonResponse(res));
  } catch (err) {
    throw classifyClientError(err);
  }
}

export { checkServerHealth } from './apiClient';

function guessMimeType(filename: string, mimeType?: string | null): string {
  if (mimeType) return mimeType;
  const lower = filename.toLowerCase();
  if (lower.endsWith('.pdf')) return 'application/pdf';
  if (lower.endsWith('.txt') || lower.endsWith('.md')) return 'text/plain';
  if (lower.endsWith('.json')) return 'application/json';
  return 'application/octet-stream';
}

/**
 * Upload a local file via expo-file-system native multipart upload.
 *
 * Expo SDK 57 WinterCG fetch rejects classic RN FormData `{ uri, name, type }`.
 * File.upload(UploadType.MULTIPART) is the supported path. Authorization must
 * be attached explicitly — uploads do not go through fetchWithAuth.
 */
async function multipartUploadFile(
  file: InstanceType<typeof import('expo-file-system').File>,
  url: string,
  resolvedMime: string
): Promise<UploadResponse> {
  const { UploadType } = await import('expo-file-system');
  const headers = await authHeadersForUpload();

  const run = async (auth: Record<string, string>) =>
    file.upload(url, {
      httpMethod: 'POST',
      uploadType: UploadType.MULTIPART,
      fieldName: 'file',
      mimeType: resolvedMime,
      headers: auth,
    });

  let result = await run(headers);
  if (result.status === 401) {
    const renewed = await refreshAccessToken();
    if (!renewed) {
      throw new Error('Session expired — please sign in again');
    }
    result = await run(await getAuthHeaders());
  }

  if (result.status < 200 || result.status >= 300) {
    let body: unknown = {};
    try {
      body = JSON.parse(result.body);
    } catch {
      body = { error: result.body || `HTTP ${result.status}` };
    }
    throw new Error(formatApiError(result.status, body, `HTTP ${result.status}`));
  }

  let parsed: unknown = {};
  try {
    parsed = JSON.parse(result.body);
  } catch {
    throw new Error('Upload failed: invalid server response');
  }
  const normalized = normalizeUploadResponse(parsed);
  if (!normalized.documentId) {
    throw new Error('Upload failed: server did not return document_id');
  }
  return normalized;
}

export async function uploadDocument(
  uri: string,
  name: string,
  mimeType?: string | null
): Promise<UploadResponse> {
  await ensureConnected();
  const url = buildApiUrl('/api/documents/upload');
  const resolvedMime = guessMimeType(name, mimeType);

  if (__DEV__) {
    console.log('[upload] selected file:', { uri, name, type: resolvedMime });
    console.log('[upload] sending request:', url);
  }

  try {
    const { File } = await import('expo-file-system');
    const file = new File(uri);

    if (__DEV__) {
      console.log('[upload] file exists:', file.exists, 'size:', file.size);
    }

    if (!file.exists) {
      throw new Error(
        `Upload failed: selected file is not readable at ${uri}`
      );
    }

    const normalized = await multipartUploadFile(file, url, resolvedMime);
    if (__DEV__) {
      console.log(
        '[upload] upload completed:',
        normalized.documentId,
        normalized.chunkCount
      );
    }
    return {
      ...normalized,
      filename: normalized.filename || name,
    };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);

    if (
      detail.startsWith('AI provider') ||
      detail.startsWith('RAG service') ||
      detail.startsWith('Document rejected') ||
      detail.startsWith('Database error') ||
      detail.startsWith('Server error') ||
      detail.startsWith('Upload failed') ||
      detail.startsWith('Rate limit') ||
      detail.includes('rate limit') ||
      detail.includes('Session expired')
    ) {
      throw err instanceof Error ? err : new Error(detail);
    }

    if (detail.includes('FormDataPart') || detail.includes('FormData')) {
      throw new Error(
        `Upload failed: unsupported file upload format.\n\n${detail}`
      );
    }

    if (
      detail.includes('Network request failed') ||
      detail.includes('Failed to fetch') ||
      detail.includes('Could not connect')
    ) {
      throw new Error(
        `Upload failed: could not reach server.\n\n${getConnectionHelpMessage()}`
      );
    }

    throw new Error(`Upload failed: ${detail}`);
  }
}

/**
 * Paste-to-ingest: backend `/api/documents/ingest` expects multipart `file`,
 * not JSON `{ text }`. Write a temp .txt and reuse the authenticated uploader.
 */
export async function ingestDocument(
  text: string,
  _strategy = 'recursive'
): Promise<{ documentId: string; chunkCount: number; filename: string }> {
  await ensureConnected();
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error('Paste text is empty');
  }

  try {
    const { File, Paths } = await import('expo-file-system');
    const filename = `pasted-${Date.now()}.txt`;
    const file = new File(Paths.cache, filename);
    file.create();
    file.write(trimmed);

    const url = buildApiUrl('/api/documents/ingest');
    const normalized = await multipartUploadFile(file, url, 'text/plain');
    try {
      file.delete();
    } catch {
      // Best-effort cache cleanup.
    }
    return {
      documentId: normalized.documentId,
      chunkCount: normalized.chunkCount,
      filename: normalized.filename || filename,
    };
  } catch (err) {
    throw classifyClientError(err);
  }
}

export interface DocumentListItem {
  documentId: string;
  source?: string;
  chunkCount: number;
  strategy?: string;
}

export async function listDocuments(): Promise<DocumentListItem[]> {
  await ensureConnected();
  try {
    const res = await fetchWithAuth('/api/documents', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    const body = await parseJsonResponse(res);
    return normalizeDocumentList(body);
  } catch (err) {
    throw classifyClientError(err);
  }
}
