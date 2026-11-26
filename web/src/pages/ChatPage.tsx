import { useEffect, useRef } from 'react';
import { MessageBubble } from '@/components/chat/MessageBubble';
import { ChatComposer } from '@/components/chat/ChatComposer';
import { BrandMark } from '@/components/common/BrandMark';
import { useAuth } from '@/contexts/AuthContext';
import { useChat } from '@/contexts/ChatContext';

export function ChatPage() {
  const { user } = useAuth();
  const { messages, streaming, error, sendMessage, stopGeneration, clearError } =
    useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);

  const firstName =
    user?.display_name?.split(/\s+/)[0] ||
    user?.email?.split('@')[0] ||
    'there';

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col">
          {messages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center px-4 text-center">
              <BrandMark size={56} />
              <h1 className="mt-6 text-3xl font-semibold tracking-tight text-[var(--text)]">
                Hello, {firstName}
              </h1>
              <p className="mt-3 max-w-md text-[var(--text-secondary)]">
                Ask engineering questions, upload documents for RAG, or probe
                agent tools through the live FastAPI backend.
              </p>
            </div>
          ) : (
            <div className="space-y-6 pb-6">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-[var(--border)] bg-[var(--bg)] px-4 py-4">
        <div className="mx-auto w-full max-w-3xl">
          {error ? (
            <div className="mb-3 flex items-start justify-between gap-3 rounded-xl bg-red-50 px-3 py-2 text-sm text-[var(--danger)] dark:bg-red-950/40">
              <span>{error}</span>
              <button
                type="button"
                className="underline"
                onClick={clearError}
              >
                Dismiss
              </button>
            </div>
          ) : null}
          <ChatComposer
            streaming={streaming}
            onSend={(text) => void sendMessage(text)}
            onStop={stopGeneration}
          />
        </div>
      </div>
    </div>
  );
}
