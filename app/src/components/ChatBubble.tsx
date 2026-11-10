import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Colors } from '../constants/colors';
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
  timestamp,
  model,
  provider,
  usage,
  streaming,
  citations,
}) => {
  if (isUser) {
    return (
      <View style={styles.userContainer}>
        <LinearGradient
          colors={[Colors.gradient.userBubbleStart, Colors.gradient.userBubbleEnd]}
          style={styles.userBubble}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <Text style={styles.userText}>{text}</Text>
        </LinearGradient>
        {timestamp && <Text style={styles.timestamp}>{timestamp}</Text>}
      </View>
    );
  }

  return (
    <View style={styles.aiContainer}>
      <View style={styles.aiBubble}>
        <Text style={styles.aiText}>{text}{streaming ? '▍' : ''}</Text>
        {(model || usage) && !streaming ? (
          <Text style={styles.metaText}>
            {provider ? `${provider}/` : ''}{model ?? ''}
            {usage ? ` · ${usage.totalTokens} tokens` : ''}
          </Text>
        ) : null}
        {citations?.length && !streaming ? (
          <SourceCitations citations={citations} />
        ) : null}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  userContainer: {
    alignItems: 'flex-end',
    marginBottom: 12,
    paddingHorizontal: 16,
  },
  aiContainer: {
    alignItems: 'flex-start',
    marginBottom: 12,
    paddingHorizontal: 16,
  },
  userBubble: {
    maxWidth: '78%',
    borderRadius: 18,
    borderBottomRightRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  aiBubble: {
    maxWidth: '85%',
    backgroundColor: Colors.surface,
    borderRadius: 18,
    borderBottomLeftRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: Colors.border.light,
    shadowColor: Colors.shadow.card,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 1,
  },
  userText: {
    fontSize: 14,
    fontWeight: '400',
    color: '#FFFFFF',
    lineHeight: 20,
  },
  aiText: {
    fontSize: 14,
    fontWeight: '400',
    color: Colors.text.primary,
    lineHeight: 20,
  },
  metaText: {
    fontSize: 10,
    color: Colors.text.muted,
    marginTop: 8,
  },
  timestamp: {
    fontSize: 10,
    color: Colors.text.muted,
    marginTop: 4,
    marginRight: 4,
  },
});
