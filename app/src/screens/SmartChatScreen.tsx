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
import { ChatHeader } from '../components/assistant/ChatHeader';
import { ChatComposer } from '../components/assistant/ChatComposer';
import { ConversationDrawer } from '../components/assistant/ConversationDrawer';
import { ChatBubble } from '../components/ChatBubble';
import { DocumentPasteModal } from '../components/DocumentPasteModal';
import { AssistantTheme } from '../constants/assistantTheme';
import type { RootStackParamList, ChatMessage } from '../types';
import {
  sendActionMessage,
  streamChatMessage,
  StreamCancelledError,
  StreamDeliveryError,
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
import {
  getConversation,
  newConversationId,
  upsertConversation,
  type StoredConversation,
} from '../services/conversationStore';

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
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [conversationId, setConversationId] = useState(() => newConversationId());
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [liveModel, setLiveModel] = useState<string | null>(null);
  const flatListRef = useRef<FlatList>(null);
  const bootstrapped = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  /** Guards rapid double-tap: one logical send ⇒ one stream POST. */
  const inFlightRef = useRef(false);
  const stickToBottomRef = useRef(true);
  const tokenRafRef = useRef<number | null>(null);
  const pendingTokenTextRef = useRef<{ id: string; text: string } | null>(null);

  const scrollToEnd = useCallback((animated = true) => {
    if (!stickToBottomRef.current) return;
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated }), 80);
  }, []);

  const appendMessage = useCallback(
    (msg: ChatMessage) => {
      stickToBottomRef.current = true;
      setMessages((prev) => [...prev, msg]);
      scrollToEnd();
    },
    [scrollToEnd]
  );

  const persist = useCallback(
    async (nextMessages: ChatMessage[], nextSession?: string) => {
      await upsertConversation({
        id: conversationId,
        sessionId: nextSession ?? sessionId,
        messages: nextMessages,
      });
    },
    [conversationId, sessionId]
  );

  useEffect(() => {
    if (messages.length === 0) return;
    void persist(messages, sessionId);
  }, [messages, sessionId, persist]);

  const retryConnection = useCallback(
    async (showSuccess = false) => {
      resetApiDiscovery();
      const url = await discoverApiBaseUrl();
      const ok = await checkServerHealth();
      if (ok) {
        setError(null);
        if (showSuccess) {
          appendMessage({
            id: `${Date.now()}-sys`,
            text: `Connected to server at ${url}`,
            isUser: false,
            timestamp: formatTime(),
          });
        }
      } else {
        setError(getConnectionHelpMessage());
      }
    },
    [appendMessage]
  );

  useEffect(() => {
    void retryConnection(false);
  }, [retryConnection]);

  const updateMessage = useCallback(
    (id: string, patch: Partial<ChatMessage>, opts?: { forceScroll?: boolean }) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
      );
      if (opts?.forceScroll || stickToBottomRef.current) {
        scrollToEnd(Boolean(opts?.forceScroll));
      }
    },
    [scrollToEnd]
  );

  const flushPendingToken = useCallback(() => {
    tokenRafRef.current = null;
    const pending = pendingTokenTextRef.current;
    if (!pending) return;
    pendingTokenTextRef.current = null;
    updateMessage(pending.id, { text: pending.text, streaming: true });
  }, [updateMessage]);

  const scheduleTokenUpdate = useCallback(
    (id: string, fullText: string) => {
      pendingTokenTextRef.current = { id, text: fullText };
      if (tokenRafRef.current != null) return;
      // Coalesce rapid token events to one UI update per frame.
      tokenRafRef.current = requestAnimationFrame(flushPendingToken) as unknown as number;
    },
    [flushPendingToken]
  );

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (tokenRafRef.current != null) {
      cancelAnimationFrame(tokenRafRef.current);
      tokenRafRef.current = null;
    }
    flushPendingToken();
    setLoading(false);
    setMessages((prev) =>
      prev.map((m) =>
        m.streaming
          ? {
              ...m,
              streaming: false,
              text: m.text || 'Generation stopped.',
            }
          : m
      )
    );
  }, [flushPendingToken]);

  const requestReply = useCallback(
    async (
      userText: string,
      history: { role: 'user' | 'assistant'; content: string }[],
      actionId?: string
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);
      stickToBottomRef.current = true;

      const aiId = `${Date.now()}-ai`;
      let partialText = '';

      try {
        if (actionId) {
          const response = await sendActionMessage(
            actionId,
            userText,
            history,
            sessionId
          );
          if (response.sessionId) setSessionId(response.sessionId);
          if (response.model) setLiveModel(response.model);
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
          sessionId,
          signal: controller.signal,
          onToken: (_token, fullText) => {
            partialText = fullText;
            scheduleTokenUpdate(aiId, fullText);
          },
          onMeta: (meta) => {
            if (meta.sessionId) setSessionId(meta.sessionId);
            if (meta.model) setLiveModel(meta.model);
            updateMessage(aiId, {
              model: meta.model,
              provider: meta.provider,
              citations: meta.citations,
            });
          },
        });

        if (tokenRafRef.current != null) {
          cancelAnimationFrame(tokenRafRef.current);
          tokenRafRef.current = null;
        }
        flushPendingToken();

        if (response.sessionId) setSessionId(response.sessionId);
        if (response.model) setLiveModel(response.model);
        updateMessage(
          aiId,
          {
            text: response.reply,
            streaming: false,
            model: response.model,
            provider: response.provider,
            usage: response.usage,
            citations: response.citations,
          },
          { forceScroll: stickToBottomRef.current }
        );
      } catch (err) {
        if (tokenRafRef.current != null) {
          cancelAnimationFrame(tokenRafRef.current);
          tokenRafRef.current = null;
        }
        flushPendingToken();

        if (
          err instanceof StreamCancelledError ||
          controller.signal.aborted
        ) {
          updateMessage(aiId, {
            text: partialText || 'Generation stopped.',
            streaming: false,
          });
          return;
        }
        // Preserve any tokens already rendered; never imply a silent retry.
        const preserved =
          (err instanceof StreamDeliveryError && err.partialText) ||
          partialText;
        const message =
          err instanceof Error ? err.message : 'Could not reach AI server';
        setError(message);
        updateMessage(aiId, {
          text: preserved
            ? `${preserved}\n\n⚠️ ${message}`
            : `⚠️ ${message}`,
          streaming: false,
          isUser: false,
        });
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setLoading(false);
        inFlightRef.current = false;
      }
    },
    [
      appendMessage,
      updateMessage,
      sessionId,
      scheduleTokenUpdate,
      flushPendingToken,
    ]
  );

  const handleSend = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      // Batch A: block concurrent duplicate UI submissions of the same send.
      if (!trimmed || loading || uploading || inFlightRef.current) return;
      inFlightRef.current = true;

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

      try {
        await requestReply(trimmed, history.slice(0, -1));
      } catch {
        inFlightRef.current = false;
      }
    },
    [appendMessage, loading, uploading, messages, requestReply]
  );

  const ingestSuccessMessage = (
    name: string,
    chunkCount: number,
    strategy: string
  ) =>
    `Document indexed: ${name}\n${chunkCount} chunks · strategy: ${strategy}\n\nYou can now ask questions about this document.`;

  const handleFileUpload = useCallback(async () => {
    setUploading(true);
    setError(null);
    try {
      appendMessage({
        id: `${Date.now()}-upload`,
        text: 'Selecting file…',
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
          text: 'Indexing pasted text…',
          isUser: true,
          timestamp: formatTime(),
        });
        const result = await pasteAndIngest(text);
        appendMessage({
          id: `${Date.now()}-paste-ok`,
          text: ingestSuccessMessage(
            result.filename || 'Pasted document',
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
        Alert.alert(
          'Indexed documents',
          'No documents indexed yet.\n\nUse + to upload a PDF, DOCX, CSV, TXT, or MD file.'
        );
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
      Alert.alert(
        'Could not load documents',
        err instanceof Error ? err.message : 'Error'
      );
    }
  }, []);

  const handleAttachmentPress = useCallback(() => {
    if (loading || uploading) return;

    Alert.alert('Add document', 'Choose an option', [
      {
        text: 'Pick file (PDF/DOCX/CSV/txt)',
        onPress: () => void handleFileUpload(),
      },
      { text: 'Paste text', onPress: () => setPasteVisible(true) },
      { text: 'View indexed docs', onPress: () => void handleShowDocuments() },
      { text: 'Cancel', style: 'cancel' },
    ]);
  }, [loading, uploading, handleFileUpload, handleShowDocuments]);

  const resetChat = useCallback(() => {
    setMessages([]);
    setSessionId(undefined);
    setConversationId(newConversationId());
    setError(null);
    setLiveModel(null);
  }, []);

  const loadConversation = useCallback((conversation: StoredConversation) => {
    setConversationId(conversation.id);
    setSessionId(conversation.sessionId);
    setMessages(conversation.messages);
    const lastModel = [...conversation.messages]
      .reverse()
      .find((m) => m.model)?.model;
    setLiveModel(lastModel ?? null);
    setError(null);
  }, []);

  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;

    const { initialMessage, actionId, conversationId: restoreId, openAttach } =
      route.params ?? {};

    if (restoreId) {
      void getConversation(restoreId).then((conv) => {
        if (conv) loadConversation(conv);
      });
      return;
    }

    if (openAttach) {
      setTimeout(() => handleAttachmentPress(), 300);
    }

    if (initialMessage) {
      void handleSend(initialMessage);
    } else if (actionId) {
      setLoading(true);
      sendActionMessage(actionId)
        .then((response) => {
          if (response.model) setLiveModel(response.model);
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
  }, [route.params, handleSend, appendMessage, handleAttachmentPress, loadConversation]);

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
      <StatusBar
        barStyle="dark-content"
        backgroundColor={AssistantTheme.background}
      />

      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <ChatHeader
          onMenuPress={() => navigation.goBack()}
          onTogglePress={() => setDrawerOpen(true)}
        />

        {error ? (
          <Pressable
            style={styles.errorBar}
            onPress={() => void retryConnection(true)}
          >
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.retryText}>Tap to retry connection</Text>
          </Pressable>
        ) : null}

        <KeyboardAvoidingView
          style={styles.keyboardView}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.chatContent}
            showsVerticalScrollIndicator={false}
            onScroll={(event) => {
              const { layoutMeasurement, contentOffset, contentSize } =
                event.nativeEvent;
              const distanceFromBottom =
                contentSize.height -
                (contentOffset.y + layoutMeasurement.height);
              stickToBottomRef.current = distanceFromBottom < 80;
            }}
            scrollEventThrottle={16}
            ListFooterComponent={
              loading ? (
                <View style={styles.typing}>
                  <ActivityIndicator
                    color={AssistantTheme.textSecondary}
                    size="small"
                  />
                  <Text style={styles.typingText}>Generating…</Text>
                </View>
              ) : null
            }
          />

          <ChatComposer
            onSend={handleSend}
            onVoicePress={handleVoicePress}
            onAttachmentPress={handleAttachmentPress}
            onStop={stopGeneration}
            showAttachment
            showVoice
            disabled={uploading}
            streaming={loading}
            uploading={uploading}
          />

          <DocumentPasteModal
            visible={pasteVisible}
            onClose={() => setPasteVisible(false)}
            onSubmit={handlePasteIngest}
            loading={uploading}
          />
        </KeyboardAvoidingView>
      </SafeAreaView>

      <ConversationDrawer
        visible={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onNewChat={resetChat}
        onSelectConversation={loadConversation}
        onOpenProfile={() => {
          setDrawerOpen(false);
          navigation.navigate('Profile');
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: AssistantTheme.background,
  },
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  chatContent: {
    paddingTop: 12,
    paddingBottom: 12,
    flexGrow: 1,
  },
  typing: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  typingText: {
    fontSize: 13,
    color: AssistantTheme.textSecondary,
    marginLeft: 8,
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
    color: AssistantTheme.textPrimary,
    fontWeight: '700',
    marginTop: 4,
  },
});
