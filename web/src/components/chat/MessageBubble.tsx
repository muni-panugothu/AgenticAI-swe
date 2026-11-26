import { Copy, Check, AlertCircle } from 'lucide-react';
import { useState } from 'react';
import type { ChatMessage } from '@/types/api';
import { MarkdownMessage } from '@/components/chat/MarkdownMessage';
import { CitationsPanel } from '@/components/chat/CitationsPanel';

export function MessageBubble({ message }: { message: ChatMessage }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const onCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div
      className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-[min(720px,92%)] ${
          isUser
            ? 'rounded-3xl bg-[var(--user-bubble)] px-4 py-3 text-[var(--text)]'
            : 'px-1 py-1'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-[15px] leading-7">
            {message.content}
          </p>
        ) : (
          <>
            {message.streaming && !message.content ? (
              <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[var(--text-muted)]" />
                Generating…
              </div>
            ) : (
              <div className="relative">
                <MarkdownMessage
                  content={message.content || '…'}
                  streaming={Boolean(message.streaming)}
                />
                {message.streaming ? (
                  <span
                    aria-hidden
                    className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[2px] animate-pulse bg-[var(--text)] align-baseline"
                  />
                ) : null}
              </div>
            )}
            {message.citations?.length ? (
              <CitationsPanel citations={message.citations} />
            ) : null}
            {message.contextUsed ? (
              <p className="mt-2 text-xs text-[var(--text-muted)]">
                Answer used retrieved document context
                {message.model ? ` · ${message.model}` : ''}
              </p>
            ) : null}
            {message.error ? (
              <p className="mt-2 flex items-center gap-1 text-sm text-[var(--danger)]">
                <AlertCircle size={14} /> {message.error}
              </p>
            ) : null}
            {!message.streaming && message.content ? (
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => void onCopy()}
                  className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-[var(--text-muted)] hover:bg-[var(--surface-muted)]"
                >
                  {copied ? <Check size={14} /> : <Copy size={14} />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
