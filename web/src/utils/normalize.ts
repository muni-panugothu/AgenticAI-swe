import type {
  ChatResponse,
  Citation,
  DocumentListItem,
  UploadResponse,
} from '@/types/api';

export function normalizeCitations(raw: unknown): Citation[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item, index) => {
    const row = (item ?? {}) as Record<string, unknown>;
    return {
      index:
        typeof row.index === 'number'
          ? row.index
          : typeof row.citation_index === 'number'
            ? row.citation_index
            : index + 1,
      documentId: String(row.documentId ?? row.document_id ?? ''),
      source:
        typeof row.source === 'string'
          ? row.source
          : typeof row.filename === 'string'
            ? row.filename
            : undefined,
      section: typeof row.section === 'string' ? row.section : undefined,
      chunkIndex:
        typeof row.chunkIndex === 'number'
          ? row.chunkIndex
          : typeof row.chunk_index === 'number'
            ? row.chunk_index
            : 0,
      excerpt: String(row.excerpt ?? row.content ?? ''),
      score: typeof row.score === 'number' ? row.score : 0,
    };
  });
}

export function normalizeChatResponse(raw: Record<string, unknown>): ChatResponse {
  return {
    reply: String(raw.reply ?? ''),
    sessionId:
      typeof raw.sessionId === 'string'
        ? raw.sessionId
        : typeof raw.session_id === 'string'
          ? raw.session_id
          : null,
    action: String(raw.action ?? 'general'),
    contextUsed: Boolean(raw.contextUsed ?? raw.context_used),
    ragSkipped: Boolean(
      raw.ragSkipped ?? raw.rag_skipped ?? true
    ),
    model: String(raw.model ?? 'unknown'),
    provider: String(raw.provider ?? 'groq'),
    citations: normalizeCitations(raw.citations),
  };
}

export function normalizeUploadResponse(raw: Record<string, unknown>): UploadResponse {
  return {
    documentId: String(raw.documentId ?? raw.document_id ?? ''),
    strategy: String(raw.strategy ?? ''),
    chunkCount: Number(raw.chunkCount ?? raw.chunk_count ?? 0),
    filename: String(raw.filename ?? raw.source ?? 'document'),
    status: typeof raw.status === 'string' ? raw.status : undefined,
    duplicate: Boolean(raw.duplicate),
  };
}

export function normalizeDocumentList(raw: unknown): DocumentListItem[] {
  const list = Array.isArray(raw)
    ? raw
    : raw && typeof raw === 'object' && Array.isArray((raw as { items?: unknown }).items)
      ? ((raw as { items: unknown[] }).items)
      : raw && typeof raw === 'object' && Array.isArray((raw as { documents?: unknown }).documents)
        ? ((raw as { documents: unknown[] }).documents)
        : [];

  return list.map((item) => {
    const row = (item ?? {}) as Record<string, unknown>;
    return {
      documentId: String(row.documentId ?? row.document_id ?? ''),
      filename:
        typeof row.filename === 'string'
          ? row.filename
          : typeof row.source === 'string'
            ? row.source
            : undefined,
      source: typeof row.source === 'string' ? row.source : undefined,
      chunkCount: Number(row.chunkCount ?? row.chunk_count ?? 0),
      strategy: typeof row.strategy === 'string' ? row.strategy : undefined,
      createdAt:
        typeof row.createdAt === 'string'
          ? row.createdAt
          : typeof row.created_at === 'string'
            ? row.created_at
            : undefined,
    };
  });
}
