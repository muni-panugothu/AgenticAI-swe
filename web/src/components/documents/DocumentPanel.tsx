import { useCallback, useEffect, useState } from 'react';
import { FileText, Trash2, RefreshCw, Loader2 } from 'lucide-react';
import {
  deleteDocument,
  listDocuments,
} from '@/services/documents/documentsApi';
import type { DocumentListItem } from '@/types/api';

export function DocumentPanel() {
  const [items, setItems] = useState<DocumentListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setItems(await listDocuments(50));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const onDelete = async (id: string) => {
    setBusyId(id);
    try {
      await deleteDocument(id);
      setItems((prev) => prev.filter((d) => d.documentId !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[var(--text)]">Documents</h2>
        <button
          type="button"
          onClick={() => void load()}
          className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--surface-muted)]"
          aria-label="Refresh documents"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
          <Loader2 size={16} className="animate-spin" /> Loading…
        </div>
      ) : error ? (
        <p className="text-sm text-[var(--danger)]">{error}</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-[var(--text-muted)]">
          No documents yet. Attach a file from the chat composer.
        </p>
      ) : (
        <ul className="space-y-2">
          {items.map((doc) => (
            <li
              key={doc.documentId}
              className="flex items-center gap-3 rounded-xl bg-[var(--surface-muted)] px-3 py-2"
            >
              <FileText size={16} className="shrink-0 text-[var(--text-secondary)]" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-[var(--text)]">
                  {doc.filename || doc.source || doc.documentId.slice(0, 16)}
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  {doc.chunkCount ?? 0} chunks
                </p>
              </div>
              <button
                type="button"
                disabled={busyId === doc.documentId}
                onClick={() => void onDelete(doc.documentId)}
                className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--surface)] hover:text-[var(--danger)]"
                aria-label="Delete document"
              >
                {busyId === doc.documentId ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Trash2 size={16} />
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
