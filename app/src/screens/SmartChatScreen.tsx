import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Text,
  Alert,
  Pressable,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RouteProp } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { Header } from '../components/Header';
import { ChatBubble } from '../components/ChatBubble';
import { AIInput } from '../components/AIInput';
import { DocumentPasteModal } from '../components/DocumentPasteModal';
import { Colors } from '../constants/colors';
import type { RootStackParamList, ChatMessage } from '../types';
import {
  sendActionMessage,
  streamChatMessage,
  checkServerHealth,
  listDocuments,
} from '../services/chatApi';
import {
  discoverApiBaseUrl,
  getConnectionHelpMessage,
  resetApiDiscovery,
} from '../services/apiClient';
import {
  pickAndUploadFile,
  pasteAndIngest,
} from '../services/attachDocument';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;
type SmartChatRoute = RouteProp<RootStackParamList, 'SmartChat'>;

function formatTime(): string {
  return new Date().toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
}

export const SmartChatScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<SmartChatRoute>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [pasteVisible, setPasteVisible] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const flatListRef = useRef<FlatList>(null);
  const bootstrapped = useRef(false);

  const scrollToEnd = useCallback(() => {
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 80);
  }, []);

  const appendMessage = useCallback(
    (msg: ChatMessage) => {
      setMessages((prev) => [...prev, msg]);
      scrollToEnd();
    },
    [scrollToEnd]
  );

  const retryConnection = useCallback(async (showSuccess = false) => {
    resetApiDiscovery();
    const url = await discoverApiBaseUrl();
    const ok = await checkServerHealth();
    if (ok) {
      setError(null);
      if (showSuccess) {
        appendMessage({
          id: `${Date.now()}-sys`,
          text: `✅ Connected to server at ${url}`,
          isUser: false,
          timestamp: formatTime(),
        });
      }
    } else {
      setError(getConnectionHelpMessage());
    }
  }, [appendMessage]);

  useEffect(() => {
    void retryConnection(false);
  }, [retryConnection]);

  const updateMessage = useCallback(
    (id: string, patch: Partial<ChatMessage>) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
      );
      scrollToEnd();
    },
    [scrollToEnd]
  );

  const requestReply = useCallback(
    async (
      userText: string,
      history: { role: 'user' | 'assistant'; content: string }[],
      actionId?: string
    ) => {
      setLoading(true);
      setError(null);

      const aiId = `${Date.now()}-ai`;

      try {
        if (actionId) {
          const response = await sendActionMessage(actionId, userText, history);
          appendMessage({
            id: aiId,
            text: response.reply,
            isUser: false,
            timestamp: formatTime(),
            model: response.model,
            provider: response.provider,
            usage: response.usage,
            citations: response.citations,
          });
          return;
        }

        appendMessage({
          id: aiId,
          text: '',
          isUser: false,
          timestamp: formatTime(),
          streaming: true,
        });

        const response = await streamChatMessage(userText, history, {
          onToken: (_token, fullText) => {
            updateMessage(aiId, { text: fullText, streaming: true });
          },
          onMeta: (meta) => {
            updateMessage(aiId, {
              model: meta.model,
              provider: meta.provider,
              citations: meta.citations,
            });
          },
        });

        updateMessage(aiId, {
          text: response.reply,
          streaming: false,
          model: response.model,
          provider: response.provider,
          usage: response.usage,
          citations: response.citations,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Could not reach AI server';
        setError(message);
        updateMessage(aiId, {
          text: `⚠️ ${message}`,
          streaming: false,
          isUser: false,
        });
      } finally {
        setLoading(false);
      }
    },
    [appendMessage, updateMessage]
  );

  const handleSend = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading || uploading) return;

      const userMessage: ChatMessage = {
        id: `${Date.now()}-user`,
        text: trimmed,
        isUser: true,
        timestamp: formatTime(),
      };

      appendMessage(userMessage);

      const history = [...messages, userMessage]
        .filter((m) => !m.text.startsWith('⚠️'))
        .map((m) => ({
          role: m.isUser ? ('user' as const) : ('assistant' as const),
          content: m.text,
        }));

      await requestReply(trimmed, history.slice(0, -1));
    },
    [appendMessage, loading, uploading, messages, requestReply]
  );

  const ingestSuccessMessage = (
    name: string,
    chunkCount: number,
    strategy: string
  ) =>
    `✅ Document indexed: ${name}\n${chunkCount} chunks · strategy: ${strategy}\n\nYou can now ask questions about this document.`;

  const handleFileUpload = useCallback(async () => {
    setUploading(true);
    setError(null);
    try {
      appendMessage({
        id: `${Date.now()}-upload`,
        text: '📎 Selecting file…',
        isUser: true,
        timestamp: formatTime(),
      });
      const uploaded = await pickAndUploadFile();
      appendMessage({
        id: `${Date.now()}-upload-ok`,
        text: ingestSuccessMessage(
          uploaded.filename,
          uploaded.chunkCount,
          uploaded.strategy
        ),
        isUser: false,
        timestamp: formatTime(),
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setError(message);
      Alert.alert('Upload failed', message);
    } finally {
      setUploading(false);
    }
  }, [appendMessage]);

  const handlePasteIngest = useCallback(
    async (text: string) => {
      setPasteVisible(false);
      setUploading(true);
      setError(null);
      try {
        appendMessage({
          id: `${Date.now()}-paste`,
          text: '📋 Indexing pasted text…',
          isUser: true,
          timestamp: formatTime(),
        });
        const result = await pasteAndIngest(text);
        appendMessage({
          id: `${Date.now()}-paste-ok`,
          text: ingestSuccessMessage(
            'Pasted document',
            result.chunkCount,
            result.strategy
          ),
          isUser: false,
          timestamp: formatTime(),
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Ingest failed';
        setError(message);
        Alert.alert('Ingest failed', message);
      } finally {
        setUploading(false);
      }
    },
    [appendMessage]
  );

  const handleShowDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      if (docs.length === 0) {
        Alert.alert('Indexed documents', 'No documents indexed yet.\n\nUse + to upload a PDF, DOCX, CSV, TXT, or MD file.');
        return;
      }
      const lines = docs
        .slice(0, 15)
        .map((d) => `• ${d.source ?? d.documentId} (${d.chunkCount} chunks)`)
        .join('\n');
      Alert.alert(
        `Indexed documents (${docs.length})`,
        `${lines}${docs.length > 15 ? '\n…' : ''}`
      );
    } catch (err) {
      Alert.alert('Could not load documents', err instanceof Error ? err.message : 'Error');
    }
  }, []);

  const handleAttachmentPress = useCallback(() => {
    if (loading || uploading) return;

    Alert.alert('Add document', 'Choose an option', [
      { text: 'Pick file (PDF/DOCX/CSV/txt)', onPress: () => void handleFileUpload() },
      { text: 'Paste text', onPress: () => setPasteVisible(true) },
      { text: 'View indexed docs', onPress: () => void handleShowDocuments() },
      { text: 'Cancel', style: 'cancel' },
    ]);
  }, [loading, uploading, handleFileUpload, handleShowDocuments]);

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;

    const { initialMessage, actionId } = route.params ?? {};
    if (initialMessage) {
      handleSend(initialMessage);
    } else if (actionId) {
      setLoading(true);
      sendActionMessage(actionId)
        .then((response) => {
          appendMessage({
            id: `${Date.now()}-ai`,
            text: response.reply,
            isUser: false,
            timestamp: formatTime(),
            model: response.model,
            provider: response.provider,
            usage: response.usage,
            citations: response.citations,
          });
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : 'Server unavailable');
        })
        .finally(() => setLoading(false));
    }
  }, [route.params, handleSend, appendMessage]);

  const handleBack = useCallback(() => {
    navigation.goBack();
  }, [navigation]);

  const handleVoicePress = useCallback(() => {
    navigation.navigate('VoiceChat');
  }, [navigation]);

  const renderMessage = useCallback(
    ({ item }: { item: ChatMessage }) => (
      <ChatBubble
        text={item.text}
        isUser={item.isUser}
        timestamp={item.isUser ? item.timestamp : undefined}
        model={item.model}
        provider={item.provider}
        usage={item.usage}
        streaming={item.streaming}
        citations={item.citations}
      />
    ),
    []
  );

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor={Colors.background} />
      <LinearGradient
        colors={[Colors.background, '#FDFBFF']}
        style={StyleSheet.absoluteFill}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
      />

      <SafeAreaView style={styles.container} edges={['top']}>
        <Header
          title="Smart Chat"
          onBack={handleBack}
          rightIcon="ellipsis-vertical"
        />

        {error ? (
          <Pressable style={styles.errorBar} onPress={() => void retryConnection(true)}>
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.retryText}>Tap to retry connection</Text>
          </Pressable>
        ) : null}

        <KeyboardAvoidingView
          style={styles.keyboardView}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
        >
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.chatContent}
            showsVerticalScrollIndicator={false}
            ListFooterComponent={
              loading ? (
                <View style={styles.typing}>
                  <ActivityIndicator color={Colors.primary.main} size="small" />
                  <Text style={styles.typingText}>AutoPilot is thinking…</Text>
                </View>
              ) : null
            }
          />

          <View style={styles.inputArea}>
            <AIInput
              placeholder="Ask me anything..."
              onSend={handleSend}
              onVoicePress={handleVoicePress}
              onAttachmentPress={handleAttachmentPress}
              showAttachment={true}
              showVoice={true}
              disabled={loading}
              uploading={uploading}
            />
          </View>

          <DocumentPasteModal
            visible={pasteVisible}
            onClose={() => setPasteVisible(false)}
            onSubmit={handlePasteIngest}
            loading={uploading}
          />
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  chatContent: {
    paddingTop: 8,
    paddingBottom: 8,
    flexGrow: 1,
  },
  inputArea: {
    paddingVertical: 10,
    paddingBottom: Platform.OS === 'ios' ? 8 : 14,
    borderTopWidth: 1,
    borderTopColor: Colors.border.subtle,
    backgroundColor: 'rgba(248, 245, 255, 0.95)',
  },
  typing: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  typingText: {
    fontSize: 13,
    color: Colors.text.secondary,
  },
  errorBar: {
    backgroundColor: '#FEE2E2',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  errorText: {
    fontSize: 11,
    color: '#B91C1C',
    lineHeight: 16,
  },
  retryText: {
    fontSize: 11,
    color: '#7C3AED',
    fontWeight: '700',
    marginTop: 4,
  },
});
