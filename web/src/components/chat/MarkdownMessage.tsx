import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Copy, Check } from 'lucide-react';
import { useState, type ReactNode } from 'react';

function CodeBlock({
  children,
  className,
}: {
  children?: ReactNode;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const text = String(children ?? '').replace(/\n$/, '');
  const language = /language-([\w-]+)/.exec(className ?? '')?.[1];

  const onCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group relative my-3 overflow-hidden rounded-xl border border-[var(--border)] bg-[#0d1117]">
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-1.5 text-xs text-white/60">
        <span>{language || 'code'}</span>
        <button
          type="button"
          onClick={() => void onCopy()}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 hover:bg-white/10"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-sm leading-6 text-[#e6edf3]">
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

export function MarkdownMessage({
  content,
  streaming = false,
}: {
  content: string;
  streaming?: boolean;
}) {
  // While streaming, skip syntax highlighting so incomplete fences/tokens
  // do not thrash the highlighter or throw on partial code blocks.
  const rehype = streaming ? [] : [rehypeHighlight];
  return (
    <div className="prose-chat max-w-none text-[15px] leading-7 text-[var(--text)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={rehype}
        components={{
          code({ className, children, ...props }) {
            const isBlock = Boolean(className) || String(children).includes('\n');
            if (!isBlock) {
              return (
                <code
                  className="rounded bg-[var(--surface-muted)] px-1.5 py-0.5 text-[0.9em]"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return <CodeBlock className={className}>{children}</CodeBlock>;
          },
          a({ href, children }) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-[var(--ring)] underline underline-offset-2"
              >
                {children}
              </a>
            );
          },
          ul({ children }) {
            return <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>;
          },
          ol({ children }) {
            return (
              <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
            );
          },
          p({ children }) {
            return <p className="mb-3 last:mb-0">{children}</p>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
