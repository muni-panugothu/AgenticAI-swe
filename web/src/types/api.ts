export interface AuthUser {
  id: string;
  email: string;
  display_name?: string | null;
  tenant_id?: string | null;
  status: 'active' | 'disabled';
  created_at?: string | null;
  terms_accepted_at?: string | null;
  privacy_accepted_at?: string | null;
  terms_version?: string | null;
  privacy_version?: string | null;
}

export interface AuthTokensResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
  refresh_token?: string | null;
}

export interface ChatHistoryItem {
  role: 'user' | 'assistant' | string;
  content: string;
}

export interface Citation {
  index: number;
  documentId: string;
  source?: string;
  section?: string;
  chunkIndex: number;
  excerpt: string;
  score: number;
}

export interface ChatResponse {
  reply: string;
  sessionId?: string | null;
  action: string;
  contextUsed: boolean;
  ragSkipped: boolean;
  model: string;
  provider: string;
  citations: Citation[];
}

export type StreamEvent =
  | {
      type: 'meta';
      sessionId?: string;
      action?: string;
      contextUsed?: boolean;
      ragSkipped?: boolean;
      model?: string;
      provider?: string;
      citations?: Citation[];
    }
  | { type: 'token'; token: string }
  | {
      type: 'done';
      sessionId?: string;
      model?: string;
      provider?: string;
      citations?: Citation[];
    }
  | { type: 'heartbeat'; ts?: number }
  | { type: 'error'; error: string; code?: string };

export interface DocumentListItem {
  documentId: string;
  filename?: string;
  source?: string;
  chunkCount?: number;
  strategy?: string;
  createdAt?: string;
}

export interface UploadResponse {
  documentId: string;
  strategy: string;
  chunkCount: number;
  filename: string;
  status?: string;
  duplicate?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  streaming?: boolean;
  error?: string;
  citations?: Citation[];
  contextUsed?: boolean;
  ragSkipped?: boolean;
  model?: string;
  provider?: string;
}
