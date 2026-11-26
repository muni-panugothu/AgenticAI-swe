import { describe, expect, it } from 'vitest';
import {
  normalizeChatResponse,
  normalizeCitations,
  normalizeUploadResponse,
} from '@/utils/normalize';

describe('normalizeCitations', () => {
  it('maps snake_case and camelCase citation fields', () => {
    const result = normalizeCitations([
      {
        document_id: 'doc1',
        chunk_index: 2,
        excerpt: 'hello',
        score: 0.9,
        filename: 'guide.pdf',
      },
    ]);
    expect(result[0]).toMatchObject({
      documentId: 'doc1',
      chunkIndex: 2,
      excerpt: 'hello',
      source: 'guide.pdf',
    });
  });
});

describe('normalizeChatResponse', () => {
  it('normalizes chat payload', () => {
    const chat = normalizeChatResponse({
      reply: 'Hi',
      sessionId: 's1',
      action: 'general',
      contextUsed: true,
      ragSkipped: false,
      model: 'm',
      provider: 'groq',
      citations: [],
    });
    expect(chat.reply).toBe('Hi');
    expect(chat.sessionId).toBe('s1');
    expect(chat.contextUsed).toBe(true);
  });
});

describe('normalizeUploadResponse', () => {
  it('reads upload fields', () => {
    expect(
      normalizeUploadResponse({
        document_id: 'd',
        chunk_count: 3,
        filename: 'a.txt',
        strategy: 'text',
      })
    ).toMatchObject({
      documentId: 'd',
      chunkCount: 3,
      filename: 'a.txt',
    });
  });
});
