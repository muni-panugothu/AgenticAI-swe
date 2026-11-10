import {
  discoverApiBaseUrl,
  buildApiUrl,
  getConnectionHelpMessage,
} from './apiClient';
import { fetchWithAuth } from './authorizedFetch';

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

interface ServerErrorBody {
  error?: string;
  code?: string;
  destination?: string;
}

function formatServerError(status: number, body: ServerErrorBody): string {
  const code = body.code ?? '';
  const dest = body.destination ? ` (→ ${body.destination})` : '';

  switch (code) {
    case 'AI_PROVIDER_ERROR':
      return `AI provider failed${dest}: ${body.error ?? 'unknown'}`;
    case 'AI_PROVIDER_AUTH':
      return 'AI provider authentication failed — check your API key in server/.env';
    case 'AI_PROVIDER_RATE_LIMIT':
      return 'AI provider rate limited — wait a moment and try again';
    case 'RAG_UNAVAILABLE':
      return `RAG service unavailable${dest}: ${body.error ?? 'Qdrant not running'}`;
    case 'MONGODB_ERROR':
      return `Database error: ${body.error ?? 'MongoDB unavailable'}`;
    default:
      return body.error ?? `Server error (HTTP ${status})`;
  }
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText } as ServerErrorBody));
    throw new Error(formatServerError(res.status, body as ServerErrorBody));
  }
  return res.json() as Promise<T>;
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
    return parseJson<ChatResponse>(res);
  } catch (err) {
    throw classifyClientError(err);
  }
}

/**
 * Stream chat tokens via SSE from POST /api/chat/stream.
 * Falls back to blocking sendChatMessage if the response body is not readable.
 */
export async function streamChatMessage(
  message: string,
  history: ChatHistoryItem[] = [],
  options: {
    action?: string;
    useRag?: boolean;
    sessionId?: string;
    onToken: (token: string, fullText: string) => void;
    onMeta?: (meta: Extract<StreamEvent, { type: 'meta' }>) => void;
    onDone?: (done: Extract<StreamEvent, { type: 'done' }>) => void;
  }
): Promise<ChatResponse> {
  await ensureConnected();

  const res = await fetchWithAuth('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      message,
      history,
      action: options.action,
      useRag: options.useRag ?? true,
      sessionId: options.sessionId,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(formatServerError(res.status, body as ServerErrorBody));
  }

  const reader = res.body?.getReader?.();
  if (!reader) {
    return sendChatMessage(message, history, options);
  }

  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';
  let meta: Extract<StreamEvent, { type: 'meta' }> | undefined;
  let doneEvt: Extract<StreamEvent, { type: 'done' }> | undefined;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() ?? '';

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      const json = line.slice(5).trim();
      if (!json) continue;
      const evt = JSON.parse(json) as StreamEvent;
      if (evt.type === 'meta') {
        meta = evt;
        options.onMeta?.(evt);
      } else if (evt.type === 'token') {
        fullText += evt.token;
        options.onToken(evt.token, fullText);
      } else if (evt.type === 'done') {
        doneEvt = evt;
        options.onDone?.(evt);
      }
    }
  }

  return {
    reply: fullText,
    sessionId: doneEvt?.sessionId ?? meta?.sessionId,
    action: meta?.action ?? 'general',
    contextUsed: meta?.contextUsed ?? false,
    ragSkipped: meta?.ragSkipped,
    model: doneEvt?.model ?? meta?.model ?? 'unknown',
    provider: doneEvt?.provider ?? meta?.provider,
    routingReason: meta?.routingReason,
    usage: doneEvt?.usage,
    estimatedCostUsd: doneEvt?.estimatedCostUsd,
    inputTokensEstimate: meta?.inputTokensEstimate,
    citations: doneEvt?.citations ?? meta?.citations,
  };
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
      body: JSON.stringify({ message, history, sessionId }),
    });
    return parseJson<ChatResponse>(res);
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
 * Expo SDK 57 replaces global fetch with WinterCG fetch, which rejects the
 * classic React Native FormData descriptor `{ uri, name, type }` and throws:
 *   "Unsupported FormDataPart implementation"
 * (see expo/src/winter/fetch/convertFormData.ts).
 *
 * Using File.upload(UploadType.MULTIPART) avoids that path entirely.
 */
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
    const { File, UploadType } = await import('expo-file-system');
    const file = new File(uri);

    if (__DEV__) {
      console.log('[upload] file exists:', file.exists, 'size:', file.size);
    }

    if (!file.exists) {
      throw new Error(
        `Upload failed: selected file is not readable at ${uri}`
      );
    }

    const result = await file.upload(url, {
      httpMethod: 'POST',
      uploadType: UploadType.MULTIPART,
      fieldName: 'file',
      mimeType: resolvedMime,
      parameters: { strategy: 'recursive' },
    });

    if (__DEV__) {
      console.log('[upload] response status:', result.status);
    }

    if (result.status < 200 || result.status >= 300) {
      let body: ServerErrorBody = {};
      try {
        body = JSON.parse(result.body) as ServerErrorBody;
      } catch {
        body = { error: result.body || `HTTP ${result.status}` };
      }
      throw new Error(formatServerError(result.status, body));
    }

    const parsed = JSON.parse(result.body) as UploadResponse;
    if (__DEV__) {
      console.log('[upload] upload completed:', parsed.documentId, parsed.chunkCount);
    }
    return parsed;
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);

    if (
      detail.startsWith('AI provider') ||
      detail.startsWith('RAG service') ||
      detail.startsWith('Database error') ||
      detail.startsWith('Server error') ||
      detail.startsWith('Upload failed')
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

export async function ingestDocument(
  text: string,
  strategy = 'recursive'
): Promise<{ documentId: string; chunkCount: number }> {
  await ensureConnected();
  try {
    const res = await fetch(buildApiUrl('/api/documents/ingest'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, strategy }),
    });
    return parseJson(res);
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
    const res = await fetch(buildApiUrl('/api/documents'), {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    const body = await parseJson<{ documents: DocumentListItem[] }>(res);
    return body.documents ?? [];
  } catch (err) {
    throw classifyClientError(err);
  }
}
