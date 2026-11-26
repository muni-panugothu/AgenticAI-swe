import React, { useCallback, useState } from 'react';
import {
  View,
  StyleSheet,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { ChatHeader } from '../components/assistant/ChatHeader';
import { HomeGreeting } from '../components/assistant/HomeGreeting';
import { HomeQuickActions } from '../components/assistant/HomeQuickActions';
import type { HomeQuickActionId } from '../components/assistant/HomeQuickActions';
import { ChatComposer } from '../components/assistant/ChatComposer';
import { ConversationDrawer } from '../components/assistant/ConversationDrawer';
import { AssistantTheme } from '../constants/assistantTheme';
import type { RootStackParamList } from '../types';
import type { StoredConversation } from '../services/conversationStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleSend = useCallback(
    (text: string) => {
      navigation.navigate('SmartChat', { initialMessage: text });
    },
    [navigation]
  );

  const handleNewChat = useCallback(() => {
    navigation.navigate('SmartChat', {});
  }, [navigation]);

  const handleSelectConversation = useCallback(
    (conversation: StoredConversation) => {
      navigation.navigate('SmartChat', { conversationId: conversation.id });
    },
    [navigation]
  );

  const handleQuickAction = useCallback(
    (id: HomeQuickActionId) => {
      if (id === 'voice') {
        navigation.navigate('VoiceChat');
        return;
      }
      if (id === 'image') {
        navigation.navigate('SmartChat', {
          initialMessage: 'Create an image or sticker for me',
        });
        return;
      }
      navigation.navigate('SmartChat', {
        initialMessage: 'Help me write or edit something',
      });
    },
    [navigation]
  );

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor={AssistantTheme.background} />
      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        <ChatHeader
          onMenuPress={() => navigation.navigate('Profile')}
          onTogglePress={() => setDrawerOpen(true)}
        />

        <KeyboardAvoidingView
          style={styles.body}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <HomeGreeting />

          <HomeQuickActions onAction={handleQuickAction} />

          <ChatComposer
            placeholder="Ask anything"
            onSend={handleSend}
            onAttachmentPress={() =>
              navigation.navigate('SmartChat', { openAttach: true })
            }
            onVoicePress={() => navigation.navigate('VoiceChat')}
          />
        </KeyboardAvoidingView>
      </SafeAreaView>

      <ConversationDrawer
        visible={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
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
  safe: {
    flex: 1,
  },
  body: {
    flex: 1,
  },
});
