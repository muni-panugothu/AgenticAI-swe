/**
 * Local conversation index for the right-side drawer.
 * Does not replace server sessionId — only caches UI history on-device.
 */

import type { ChatMessage } from '../types';

export interface StoredConversation {
  id: string;
  sessionId?: string;
  title: string;
  preview: string;
  updatedAt: string;
  messages: ChatMessage[];
}

const FILE = 'agentic_conversations.json';
const MAX = 40;

async function readAll(): Promise<StoredConversation[]> {
  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, FILE);
    if (!file.exists) return [];
    const parsed = JSON.parse(await file.text()) as { items?: StoredConversation[] };
    return Array.isArray(parsed.items) ? parsed.items : [];
  } catch {
    return [];
  }
}

async function writeAll(items: StoredConversation[]): Promise<void> {
  try {
    const { File, Paths } = await import('expo-file-system');
    const file = new File(Paths.document, FILE);
    if (!file.exists) file.create();
    file.write(JSON.stringify({ items: items.slice(0, MAX) }));
  } catch (err) {
    if (__DEV__) console.warn('[conversationStore] write failed', err);
  }
}

function titleFromMessages(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.isUser && m.text.trim());
  if (!firstUser) return 'New chat';
  const t = firstUser.text.trim().replace(/\s+/g, ' ');
  return t.length > 42 ? `${t.slice(0, 42)}…` : t;
}

export async function listConversations(): Promise<StoredConversation[]> {
  const items = await readAll();
  return items.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export async function getConversation(
  id: string
): Promise<StoredConversation | null> {
  const items = await readAll();
  return items.find((c) => c.id === id) ?? null;
}

export async function upsertConversation(input: {
  id: string;
  sessionId?: string;
  messages: ChatMessage[];
}): Promise<void> {
  if (!input.messages.length) return;
  const items = await readAll();
  const preview =
    [...input.messages].reverse().find((m) => m.text.trim())?.text.trim() ?? '';
  const next: StoredConversation = {
    id: input.id,
    sessionId: input.sessionId,
    title: titleFromMessages(input.messages),
    preview: preview.slice(0, 80),
    updatedAt: new Date().toISOString(),
    messages: input.messages,
  };
  const without = items.filter((c) => c.id !== input.id);
  await writeAll([next, ...without]);
}

export async function deleteConversation(id: string): Promise<void> {
  const items = await readAll();
  await writeAll(items.filter((c) => c.id !== id));
}

export function newConversationId(): string {
  return `conv_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
