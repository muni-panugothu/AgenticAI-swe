import { apiFetch, apiJson } from '@/services/api/client';
import type { DocumentListItem, UploadResponse } from '@/types/api';
import { formatApiError } from '@/utils/errors';
import {
  normalizeDocumentList,
  normalizeUploadResponse,
} from '@/utils/normalize';

export const ACCEPTED_UPLOAD_EXTENSIONS =
  '.txt,.md,.markdown,.pdf,.docx,.html,.htm,.csv';

export async function listDocuments(limit = 20): Promise<DocumentListItem[]> {
  const raw = await apiJson<unknown>(
    `/api/documents?limit=${encodeURIComponent(String(limit))}`,
    { method: 'GET' }
  );
  return normalizeDocumentList(raw);
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await apiFetch('/api/documents/upload', {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(formatApiError(res.status, body, res.statusText));
  }

  const raw = (await res.json()) as Record<string, unknown>;
  return normalizeUploadResponse(raw);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(formatApiError(res.status, body, res.statusText));
  }
}
