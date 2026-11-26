import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { AssistantTheme } from '../constants/assistantTheme';
import { SourceCitations, type CitationItem } from './SourceCitations';

interface ChatBubbleProps {
  text: string;
  isUser: boolean;
  timestamp?: string;
  model?: string;
  provider?: string;
  usage?: { totalTokens: number };
  streaming?: boolean;
  citations?: CitationItem[];
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({
  text,
  isUser,
  model,
  provider,
  usage,
  streaming,
  citations,
}) => {
  if (isUser) {
    return (
      <View style={styles.userContainer}>
        <View style={styles.userBubble}>
          <Text style={styles.userText}>{text}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.aiContainer}>
      <Text style={styles.aiText}>
        {text}
        {streaming ? ' ▍' : ''}
      </Text>
      {(model || usage) && !streaming ? (
        <Text style={styles.metaText}>
          {provider ? `${provider}/` : ''}
          {model ?? ''}
          {usage ? ` · ${usage.totalTokens} tokens` : ''}
        </Text>
      ) : null}
      {citations?.length && !streaming ? (
        <SourceCitations citations={citations} />
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  userContainer: {
    alignItems: 'flex-end',
    marginBottom: 14,
    paddingHorizontal: 16,
  },
  aiContainer: {
    alignItems: 'flex-start',
    marginBottom: 18,
    paddingHorizontal: 18,
  },
  userBubble: {
    maxWidth: '82%',
    backgroundColor: AssistantTheme.userBubble,
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 11,
  },
  userText: {
    fontSize: 15,
    fontWeight: '400',
    color: AssistantTheme.textPrimary,
    lineHeight: 22,
  },
  aiText: {
    fontSize: 15,
    fontWeight: '400',
    color: AssistantTheme.textPrimary,
    lineHeight: 23,
    maxWidth: '95%',
  },
  metaText: {
    fontSize: 11,
    color: AssistantTheme.textMuted,
    marginTop: 8,
  },
});
