import type { Citation } from '@/types/api';
import { FileText } from 'lucide-react';

export function CitationsPanel({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;

  return (
    <div className="mt-3 rounded-xl border border-[var(--border)] bg-[var(--surface-muted)] p-3">
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
        Sources
      </p>
      <ul className="space-y-2">
        {citations.map((c) => (
          <li
            key={`${c.documentId}-${c.chunkIndex}-${c.index}`}
            className="rounded-lg bg-[var(--surface)] px-3 py-2 text-sm"
          >
            <div className="mb-1 flex items-center gap-2 text-[var(--text-secondary)]">
              <FileText size={14} />
              <span className="font-medium text-[var(--text)]">
                [{c.index}] {c.source || c.documentId.slice(0, 12)}
              </span>
              {c.section ? (
                <span className="text-xs text-[var(--text-muted)]">
                  · {c.section}
                </span>
              ) : null}
            </div>
            {c.excerpt ? (
              <p className="text-[var(--text-secondary)] line-clamp-3">
                {c.excerpt}
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
