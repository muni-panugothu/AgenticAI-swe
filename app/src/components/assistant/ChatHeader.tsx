import React from 'react';
import { View, Text, Pressable, StyleSheet, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AssistantTheme } from '../../constants/assistantTheme';

interface ChatHeaderProps {
  onMenuPress?: () => void;
  onTogglePress: () => void;
  onUpgradePress?: () => void;
  planLabel?: string;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  onMenuPress,
  onTogglePress,
  onUpgradePress,
  planLabel = 'Free plan',
}) => {
  const handleUpgrade = () => {
    if (onUpgradePress) {
      onUpgradePress();
      return;
    }
    Alert.alert(
      'Upgrade',
      'Paid plans are not configured in this build. Contact support to enable billing.'
    );
  };

  return (
    <View style={styles.row}>
      <Pressable
        onPress={onMenuPress}
        hitSlop={10}
        style={styles.iconBtn}
        accessibilityRole="button"
        accessibilityLabel="Open menu"
      >
        <Ionicons name="menu-outline" size={26} color={AssistantTheme.textPrimary} />
      </Pressable>

      <Pressable
        onPress={handleUpgrade}
        style={styles.planPill}
        accessibilityRole="button"
        accessibilityLabel={`${planLabel}, Upgrade`}
      >
        <Text style={styles.planText}>
          {planLabel}
          <Text style={styles.planDot}> • </Text>
          <Text style={styles.upgrade}>Upgrade</Text>
        </Text>
      </Pressable>

      <Pressable
        onPress={onTogglePress}
        hitSlop={10}
        style={styles.toggleBtn}
        accessibilityRole="button"
        accessibilityLabel="Open conversations panel"
      >
        <Ionicons
          name="eye-outline"
          size={22}
          color={AssistantTheme.textPrimary}
        />
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 8,
    minHeight: 48,
  },
  iconBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleBtn: {
    width: 40,
    height: 40,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: AssistantTheme.headerPill,
  },
  planPill: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: AssistantTheme.headerPill,
  },
  planText: {
    fontSize: 13,
    color: AssistantTheme.textPrimary,
    fontWeight: '500',
  },
  planDot: {
    color: AssistantTheme.textSecondary,
  },
  upgrade: {
    textDecorationLine: 'underline',
    fontWeight: '600',
  },
});
