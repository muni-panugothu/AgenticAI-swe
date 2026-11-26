/**
 * Map backend FastAPI / RAG payloads into the shapes the Expo client expects.
 * Pure helpers — no I/O — so they stay easy to unit-test offline.
 */

export interface NormalizedCitation {
  index: number;
  documentId: string;
  source?: string;
  section?: string;
  chunkIndex: number;
  excerpt: string;
  score: number;
}

export interface NormalizedUpload {
  documentId: string;
  strategy: string;
  chunkCount: number;
  filename: string;
  status?: string;
  duplicate?: boolean;
}

export interface NormalizedDocumentListItem {
  documentId: string;
  source?: string;
  chunkCount: number;
  strategy?: string;
}

/** Unwrap FastAPI `{ detail: { error, code } }` or top-level error bodies. */
export function extractErrorBody(raw: unknown): {
  error?: string;
  code?: string;
  destination?: string;
  retry_after_seconds?: number;
} {
  if (!raw || typeof raw !== 'object') {
    return {};
  }
  const body = raw as Record<string, unknown>;
  const detail = body.detail;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const d = detail as Record<string, unknown>;
    return {
      error: typeof d.error === 'string' ? d.error : undefined,
      code: typeof d.code === 'string' ? d.code : undefined,
      destination: typeof d.destination === 'string' ? d.destination : undefined,
      retry_after_seconds:
        typeof d.retry_after_seconds === 'number'
          ? d.retry_after_seconds
          : undefined,
    };
  }
  // Custom validation handler already uses top-level { error, code }.
  return {
    error: typeof body.error === 'string' ? body.error : undefined,
    code: typeof body.code === 'string' ? body.code : undefined,
    destination:
      typeof body.destination === 'string' ? body.destination : undefined,
    retry_after_seconds:
      typeof body.retry_after_seconds === 'number'
        ? body.retry_after_seconds
        : undefined,
  };
}

export function formatApiError(
  status: number,
  raw: unknown,
  statusText = ''
): string {
  const body = extractErrorBody(raw);
  const code = body.code ?? '';
  const dest = body.destination ? ` (→ ${body.destination})` : '';
  const retry =
    typeof body.retry_after_seconds === 'number'
      ? ` Retry after ${body.retry_after_seconds}s.`
      : '';

  switch (code) {
    case 'AI_PROVIDER_ERROR':
      return `AI provider failed${dest}: ${body.error ?? 'unknown'}`;
    case 'AI_PROVIDER_AUTH':
      return 'AI provider authentication failed — check server configuration';
    case 'AI_PROVIDER_RATE_LIMIT':
    case 'CHAT_RATE_LIMIT_EXCEEDED':
    case 'UPLOAD_RATE_LIMIT_EXCEEDED':
      return `${body.error ?? 'Rate limit exceeded'}.${retry}`;
    case 'CHAT_RATE_LIMIT_UNAVAILABLE':
    case 'UPLOAD_RATE_LIMIT_UNAVAILABLE':
      return body.error ?? 'Service temporarily unavailable';
    case 'RAG_UNAVAILABLE':
    case 'RAG_RETRIEVAL_FAILED':
    case 'RAG_STORE_UNAVAILABLE':
    case 'RAG_INGESTION_FAILED':
      return `RAG service unavailable${dest}: ${body.error ?? 'unavailable'}`;
    case 'DOCUMENT_REJECTED':
      return `Document rejected: ${body.error ?? 'invalid document'}`;
    case 'MONGODB_ERROR':
      return `Database error: ${body.error ?? 'MongoDB unavailable'}`;
    case 'VALIDATION_ERROR':
      return body.error ?? 'Validation failed';
    case 'UNAUTHORIZED':
    case 'INVALID_CREDENTIALS':
      return body.error ?? 'Authentication required';
    default:
      if (status === 401) {
        return body.error ?? 'Session expired — please sign in again';
      }
      if (status === 403) {
        return body.error ?? 'Not allowed';
      }
      if (status === 429) {
        return `${body.error ?? 'Too many requests'}.${retry}`;
      }
      if (status >= 500) {
        return body.error ?? `Server error (HTTP ${status})`;
      }
      return body.error ?? (statusText || `Server error (HTTP ${status})`);
  }
}

export function normalizeCitation(
  raw: Record<string, unknown>,
  fallbackIndex: number
): NormalizedCitation {
  const documentId = String(
    raw.documentId ?? raw.document_id ?? raw.chunk_id ?? `doc-${fallbackIndex}`
  );
  const chunkIndex = Number(raw.chunkIndex ?? raw.chunk_index ?? 0) || 0;
  const score = Number(raw.score ?? 0) || 0;
  const filename =
    typeof raw.filename === 'string'
      ? raw.filename
      : typeof raw.source === 'string'
        ? raw.source
        : undefined;
  const start = raw.start_line ?? raw.startLine;
  const end = raw.end_line ?? raw.endLine;
  const section =
    typeof raw.section === 'string'
      ? raw.section
      : start != null && end != null
        ? `lines ${start}–${end}`
        : undefined;
  const excerpt =
    typeof raw.excerpt === 'string'
      ? raw.excerpt
      : typeof raw.chunk_id === 'string'
        ? String(raw.chunk_id)
        : '';

  return {
    index:
      typeof raw.index === 'number' ? raw.index : fallbackIndex + 1,
    documentId,
    source: filename,
    section,
    chunkIndex,
    excerpt,
    score,
  };
}

export function normalizeCitations(raw: unknown): NormalizedCitation[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
    .map((item, i) => normalizeCitation(item, i));
}

export function normalizeUploadResponse(raw: unknown): NormalizedUpload {
  const body = (raw && typeof raw === 'object' ? raw : {}) as Record<
    string,
    unknown
  >;
  return {
    documentId: String(body.documentId ?? body.document_id ?? ''),
    strategy: String(body.strategy ?? 'default'),
    chunkCount: Number(body.chunkCount ?? body.chunk_count ?? 0) || 0,
    filename: String(body.filename ?? body.source ?? 'document'),
    status: typeof body.status === 'string' ? body.status : undefined,
    duplicate: Boolean(body.duplicate),
  };
}

export function normalizeDocumentList(raw: unknown): NormalizedDocumentListItem[] {
  const body = (raw && typeof raw === 'object' ? raw : {}) as Record<
    string,
    unknown
  >;
  const docs = Array.isArray(body.documents)
    ? body.documents
    : Array.isArray(raw)
      ? raw
      : [];
  return docs
    .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
    .map((item) => ({
      documentId: String(item.documentId ?? item.document_id ?? ''),
      source:
        typeof item.filename === 'string'
          ? item.filename
          : typeof item.source === 'string'
            ? item.source
            : undefined,
      chunkCount: Number(item.chunkCount ?? item.chunk_count ?? 0) || 0,
      strategy:
        typeof item.strategy === 'string' ? item.strategy : undefined,
    }));
}
