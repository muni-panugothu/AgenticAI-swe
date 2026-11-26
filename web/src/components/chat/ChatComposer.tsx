import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from 'react';
import { ArrowUp, Paperclip, Square, Loader2 } from 'lucide-react';
import {
  ACCEPTED_UPLOAD_EXTENSIONS,
  uploadDocument,
} from '@/services/documents/documentsApi';

interface ChatComposerProps {
  disabled?: boolean;
  streaming?: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onUploaded?: (filename: string) => void;
}

export function ChatComposer({
  disabled,
  streaming,
  onSend,
  onStop,
  onUploaded,
}: ChatComposerProps) {
  const [text, setText] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadNote, setUploadNote] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = '0px';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const value = text.trim();
    if (!value || disabled || streaming || uploading) return;
    onSend(value);
    setText('');
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const onFile = async (file: File | null) => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setUploadNote(null);
    try {
      const result = await uploadDocument(file);
      setUploadNote(
        result.duplicate
          ? `Already indexed: ${result.filename}`
          : `Uploaded ${result.filename} (${result.chunkCount} chunks)`
      );
      onUploaded?.(result.filename);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  return (
    <form onSubmit={submit} className="w-full">
      {(uploadNote || uploadError) && (
        <div
          className={`mb-2 rounded-xl px-3 py-2 text-sm ${
            uploadError
              ? 'bg-red-50 text-[var(--danger)] dark:bg-red-950/40'
              : 'bg-[var(--surface-muted)] text-[var(--text-secondary)]'
          }`}
        >
          {uploadError || uploadNote}
        </div>
      )}

      <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask anything about your code, docs, or engineering tasks…"
          rows={1}
          disabled={disabled || uploading}
          className="max-h-[180px] min-h-[52px] w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[15px] leading-6 text-[var(--text)] outline-none placeholder:text-[var(--text-muted)]"
        />

        <div className="flex items-center justify-between px-2 pb-2">
          <div className="flex items-center gap-1">
            <input
              ref={fileRef}
              type="file"
              className="hidden"
              accept={ACCEPTED_UPLOAD_EXTENSIONS}
              onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={disabled || uploading || streaming}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full text-[var(--text-secondary)] hover:bg-[var(--surface-muted)] disabled:opacity-40"
              aria-label="Attach document"
              title="Upload .txt .md .pdf .docx .html .csv"
            >
              {uploading ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Paperclip size={18} />
              )}
            </button>
          </div>

          {streaming ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex h-10 items-center gap-2 rounded-full bg-[var(--accent)] px-4 text-sm font-medium text-[var(--bg)]"
            >
              <Square size={14} fill="currentColor" />
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!text.trim() || disabled || uploading}
              className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-[var(--accent)] text-[var(--bg)] disabled:opacity-35"
              aria-label="Send message"
            >
              <ArrowUp size={18} />
            </button>
          )}
        </div>
      </div>
      <p className="mt-2 text-center text-xs text-[var(--text-muted)]">
        Enter to send · Shift+Enter for new line
      </p>
    </form>
  );
}
