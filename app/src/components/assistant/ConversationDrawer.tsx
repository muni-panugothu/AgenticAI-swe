import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  Animated,
  Dimensions,
  ScrollView,
  Modal,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { AssistantTheme } from '../../constants/assistantTheme';
import {
  listConversations,
  deleteConversation,
  type StoredConversation,
} from '../../services/conversationStore';

interface ConversationDrawerProps {
  visible: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectConversation: (conversation: StoredConversation) => void;
  onOpenProfile?: () => void;
}

const SCREEN_W = Dimensions.get('window').width;
const DRAWER_W = Math.min(SCREEN_W * 0.82, 340);

export const ConversationDrawer: React.FC<ConversationDrawerProps> = ({
  visible,
  onClose,
  onNewChat,
  onSelectConversation,
  onOpenProfile,
}) => {
  const insets = useSafeAreaInsets();
  const slide = useRef(new Animated.Value(DRAWER_W)).current;
  const fade = useRef(new Animated.Value(0)).current;
  const [items, setItems] = useState<StoredConversation[]>([]);
  const [mounted, setMounted] = useState(visible);

  useEffect(() => {
    if (visible) {
      setMounted(true);
      void listConversations().then(setItems);
      Animated.parallel([
        Animated.timing(fade, {
          toValue: 1,
          duration: 220,
          useNativeDriver: true,
        }),
        Animated.timing(slide, {
          toValue: 0,
          duration: 260,
          useNativeDriver: true,
        }),
      ]).start();
    } else if (mounted) {
      Animated.parallel([
        Animated.timing(fade, {
          toValue: 0,
          duration: 180,
          useNativeDriver: true,
        }),
        Animated.timing(slide, {
          toValue: DRAWER_W,
          duration: 220,
          useNativeDriver: true,
        }),
      ]).start(({ finished }) => {
        if (finished) setMounted(false);
      });
    }
  }, [visible, fade, slide, mounted]);

  if (!mounted) return null;

  return (
    <Modal transparent visible={mounted} animationType="none" onRequestClose={onClose}>
      <View style={styles.root}>
        <Animated.View style={[styles.overlay, { opacity: fade }]}>
          <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
        </Animated.View>

        <Animated.View
          style={[
            styles.panel,
            {
              width: DRAWER_W,
              paddingTop: insets.top + 8,
              paddingBottom: insets.bottom + 12,
              transform: [{ translateX: slide }],
            },
          ]}
        >
          <View style={styles.panelHeader}>
            <Text style={styles.panelTitle}>Conversations</Text>
            <Pressable onPress={onClose} hitSlop={10} style={styles.closeBtn}>
              <Ionicons name="close" size={22} color={AssistantTheme.textPrimary} />
            </Pressable>
          </View>

          <Pressable
            style={styles.newChatBtn}
            onPress={() => {
              onNewChat();
              onClose();
            }}
          >
            <Ionicons name="create-outline" size={20} color={AssistantTheme.textPrimary} />
            <Text style={styles.newChatText}>New chat</Text>
          </Pressable>

          <ScrollView
            style={styles.list}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
          >
            {items.length === 0 ? (
              <Text style={styles.empty}>No recent conversations yet.</Text>
            ) : (
              items.map((item) => (
                <Pressable
                  key={item.id}
                  style={styles.item}
                  onPress={() => {
                    onSelectConversation(item);
                    onClose();
                  }}
                  onLongPress={() => {
                    void deleteConversation(item.id).then(() =>
                      listConversations().then(setItems)
                    );
                  }}
                >
                  <Ionicons
                    name="chatbubble-ellipses-outline"
                    size={18}
                    color={AssistantTheme.textSecondary}
                  />
                  <View style={styles.itemTextWrap}>
                    <Text style={styles.itemTitle} numberOfLines={1}>
                      {item.title}
                    </Text>
                    <Text style={styles.itemPreview} numberOfLines={1}>
                      {item.preview}
                    </Text>
                  </View>
                </Pressable>
              ))
            )}
          </ScrollView>

          {onOpenProfile ? (
            <Pressable style={styles.footerBtn} onPress={onOpenProfile}>
              <Ionicons
                name="person-circle-outline"
                size={22}
                color={AssistantTheme.textPrimary}
              />
              <Text style={styles.footerText}>Account</Text>
            </Pressable>
          ) : null}
        </Animated.View>
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: AssistantTheme.overlay,
  },
  panel: {
    height: '100%',
    backgroundColor: AssistantTheme.surface,
    paddingHorizontal: 16,
    borderTopLeftRadius: 18,
    borderBottomLeftRadius: 18,
  },
  panelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  panelTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: AssistantTheme.textPrimary,
  },
  closeBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  newChatBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderRadius: 12,
    backgroundColor: AssistantTheme.headerPill,
    marginBottom: 12,
  },
  newChatText: {
    marginLeft: 10,
    fontSize: 15,
    fontWeight: '600',
    color: AssistantTheme.textPrimary,
  },
  list: { flex: 1 },
  listContent: { paddingBottom: 20 },
  empty: {
    color: AssistantTheme.textMuted,
    fontSize: 13,
    paddingVertical: 20,
    paddingHorizontal: 4,
  },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 10,
  },
  itemTextWrap: {
    flex: 1,
    marginLeft: 10,
  },
  itemTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: AssistantTheme.textPrimary,
  },
  itemPreview: {
    fontSize: 12,
    color: AssistantTheme.textMuted,
    marginTop: 2,
  },
  footerBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: AssistantTheme.border,
  },
  footerText: {
    marginLeft: 10,
    fontSize: 15,
    fontWeight: '500',
    color: AssistantTheme.textPrimary,
  },
});

