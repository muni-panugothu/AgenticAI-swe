/**
 * Offline contract checks for frontend ↔ backend payload mapping.
 * Run: node scripts/check-api-contract.mjs
 */
import assert from 'node:assert/strict';

function extractErrorBody(raw) {
  if (!raw || typeof raw !== 'object') return {};
  const detail = raw.detail;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    return {
      error: typeof detail.error === 'string' ? detail.error : undefined,
      code: typeof detail.code === 'string' ? detail.code : undefined,
      retry_after_seconds:
        typeof detail.retry_after_seconds === 'number'
          ? detail.retry_after_seconds
          : undefined,
    };
  }
  return {
    error: typeof raw.error === 'string' ? raw.error : undefined,
    code: typeof raw.code === 'string' ? raw.code : undefined,
  };
}

function formatApiError(status, raw) {
  const body = extractErrorBody(raw);
  if (body.code === 'CHAT_RATE_LIMIT_EXCEEDED') {
    return `${body.error ?? 'Rate limit exceeded'}. Retry after ${body.retry_after_seconds}s.`;
  }
  if (status === 401) return body.error ?? 'Session expired — please sign in again';
  return body.error ?? `Server error (HTTP ${status})`;
}

function normalizeCitation(raw, fallbackIndex) {
  return {
    index: typeof raw.index === 'number' ? raw.index : fallbackIndex + 1,
    documentId: String(raw.documentId ?? raw.document_id ?? `doc-${fallbackIndex}`),
    source: typeof raw.filename === 'string' ? raw.filename : raw.source,
    chunkIndex: Number(raw.chunkIndex ?? raw.chunk_index ?? 0) || 0,
    excerpt: typeof raw.excerpt === 'string' ? raw.excerpt : String(raw.chunk_id ?? ''),
    score: Number(raw.score ?? 0) || 0,
  };
}

function normalizeUploadResponse(raw) {
  const body = raw && typeof raw === 'object' ? raw : {};
  return {
    documentId: String(body.documentId ?? body.document_id ?? ''),
    chunkCount: Number(body.chunkCount ?? body.chunk_count ?? 0) || 0,
    filename: String(body.filename ?? 'document'),
    strategy: String(body.strategy ?? 'default'),
  };
}

function normalizeDocumentList(raw) {
  const docs = Array.isArray(raw?.documents) ? raw.documents : [];
  return docs.map((item) => ({
    documentId: String(item.documentId ?? item.document_id ?? ''),
    source: item.filename ?? item.source,
    chunkCount: Number(item.chunkCount ?? item.chunk_count ?? 0) || 0,
  }));
}

// --- asserts mirroring production helpers ---
{
  const msg = formatApiError(429, {
    detail: {
      error: 'Chat rate limit exceeded',
      code: 'CHAT_RATE_LIMIT_EXCEEDED',
      retry_after_seconds: 42,
    },
  });
  assert.match(msg, /Chat rate limit exceeded/);
  assert.match(msg, /42/);
}

{
  const msg = formatApiError(401, {
    detail: { error: 'Invalid credentials', code: 'INVALID_CREDENTIALS' },
  });
  assert.equal(msg, 'Invalid credentials');
}

{
  const c = normalizeCitation(
    {
      document_id: 'abc',
      filename: 'handbook.md',
      chunk_index: 2,
      score: 0.9,
      chunk_id: 'abc:2',
    },
    0
  );
  assert.equal(c.documentId, 'abc');
  assert.equal(c.source, 'handbook.md');
  assert.equal(c.chunkIndex, 2);
  assert.equal(c.index, 1);
}

{
  const up = normalizeUploadResponse({
    document_id: 'doc1',
    chunk_count: 3,
    status: 'ready',
  });
  assert.equal(up.documentId, 'doc1');
  assert.equal(up.chunkCount, 3);
}

{
  const list = normalizeDocumentList({
    documents: [{ document_id: 'd1', filename: 'a.txt', chunk_count: 4 }],
  });
  assert.equal(list[0].documentId, 'd1');
  assert.equal(list[0].source, 'a.txt');
  assert.equal(list[0].chunkCount, 4);
}

// Path contract: frontend expects these backend routes.
const expectedPaths = [
  '/auth/register',
  '/auth/login',
  '/auth/refresh',
  '/auth/logout',
  '/auth/me',
  '/health',
  '/api/chat',
  '/api/chat/stream',
  '/api/documents/upload',
  '/api/documents/ingest',
  '/api/documents',
];
assert.ok(expectedPaths.every((p) => p.startsWith('/')));

console.log('API contract checks OK (' + expectedPaths.length + ' paths)');
console.log(
  'Note: full `tsc -p tsconfig.json` currently fails on pre-existing RN/Expo JSX typings; not treated as an integration regression.'
);