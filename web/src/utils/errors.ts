export function extractErrorBody(raw: unknown): {
  error?: string;
  code?: string;
  details?: unknown;
  retry_after_seconds?: number;
} {
  if (!raw || typeof raw !== 'object') return {};
  const body = raw as Record<string, unknown>;
  const detail = body.detail;

  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const d = detail as Record<string, unknown>;
    return {
      error: typeof d.error === 'string' ? d.error : undefined,
      code: typeof d.code === 'string' ? d.code : undefined,
      details: d.details,
      retry_after_seconds:
        typeof d.retry_after_seconds === 'number'
          ? d.retry_after_seconds
          : undefined,
    };
  }

  if (typeof detail === 'string') {
    return { error: detail };
  }

  return {
    error: typeof body.error === 'string' ? body.error : undefined,
    code: typeof body.code === 'string' ? body.code : undefined,
    details: body.details,
    retry_after_seconds:
      typeof body.retry_after_seconds === 'number'
        ? body.retry_after_seconds
        : undefined,
  };
}

export function formatValidationDetails(details: unknown): string | null {
  if (!Array.isArray(details) || details.length === 0) return null;
  const parts: string[] = [];
  for (const item of details) {
    if (!item || typeof item !== 'object') continue;
    const row = item as { msg?: string };
    const msg = typeof row.msg === 'string' ? row.msg : null;
    if (!msg) continue;
    const clean = msg.replace(/^Value error,\s*/i, '').trim();
    if (clean && !parts.includes(clean)) parts.push(clean);
  }
  return parts.length ? parts.join('. ') : null;
}

export function formatApiError(
  status: number,
  raw: unknown,
  statusText = ''
): string {
  const body = extractErrorBody(raw);
  const fromDetails = formatValidationDetails(body.details);
  const retry =
    typeof body.retry_after_seconds === 'number'
      ? ` Retry after ${body.retry_after_seconds}s.`
      : '';

  switch (body.code) {
    case 'CHAT_RATE_LIMIT_EXCEEDED':
    case 'UPLOAD_RATE_LIMIT_EXCEEDED':
      return `${body.error ?? 'Rate limit exceeded'}.${retry}`;
    case 'DOCUMENT_REJECTED':
      return `Document rejected: ${body.error ?? 'invalid document'}`;
    case 'RAG_RETRIEVAL_FAILED':
    case 'RAG_STORE_UNAVAILABLE':
    case 'RAG_INGESTION_FAILED':
      return body.error ?? 'Document service unavailable';
    case 'INVALID_CREDENTIALS':
      return 'Invalid email or password';
    case 'EMAIL_EXISTS':
      return 'An account with that email already exists';
    case 'MISSING_REFRESH':
    case 'INVALID_REFRESH':
      return 'Session expired. Please sign in again.';
    default:
      break;
  }

  return (
    body.error ??
    fromDetails ??
    (statusText ? `HTTP ${status}: ${statusText}` : `HTTP ${status}`)
  );
}
