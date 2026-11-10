export type RootStackParamList = {
  MainTabs: undefined;
  VoiceChat: undefined;
  SmartChat: { initialMessage?: string; actionId?: string } | undefined;
  Profile: undefined;
  Login: undefined;
  Register: undefined;
};

export type BottomTabParamList = {
  Home: undefined;
  History: undefined;
  AICreate: undefined;
  Files: undefined;
  Profile: undefined;
};

export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
  image?: string;
  model?: string;
  provider?: string;
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  streaming?: boolean;
  citations?: {
    index: number;
    documentId: string;
    source?: string;
    section?: string;
    excerpt?: string;
  }[];
}
