import React, { useState } from 'react';
import {
  Pressable,
  Text,
  View,
  StyleSheet,
} from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import type { ComponentProps } from 'react';
import type { QuickAction } from '../constants/actions';

type IconName = ComponentProps<typeof MaterialCommunityIcons>['name'];

interface QuickActionCardProps {
  action: QuickAction;
  onPress?: () => void;
}

/**
 * White card — icon left, title + subtitle right (parallel row).
 */
export function QuickActionCard({ action, onPress }: QuickActionCardProps) {
  const [pressed, setPressed] = useState(false);

  return (
    <Pressable
      onPress={onPress}
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
      accessibilityRole="button"
      accessibilityLabel={action.title}
      style={[styles.card, pressed && styles.cardPressed]}
    >
      <View style={styles.row}>
        <View style={styles.iconWrap}>
          <MaterialCommunityIcons
            name={action.icon as IconName}
            size={20}
            color="#7C3AED"
          />
        </View>

        <View style={styles.textCol}>
          <Text style={styles.title}>{action.title}</Text>
          <Text style={styles.subtitle}>{action.description}</Text>
        </View>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    paddingVertical: 14,
    paddingHorizontal: 12,
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 14,
    elevation: 4,
    minHeight: 84,
    justifyContent: 'center',
  },
  cardPressed: {
    backgroundColor: '#FAFAFF',
    transform: [{ scale: 0.98 }],
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: '#F3EEFF',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
    flexShrink: 0,
  },
  textCol: {
    flex: 1,
    flexShrink: 1,
  },
  title: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1A1A2E',
    marginBottom: 2,
  },
  subtitle: {
    fontSize: 11,
    fontWeight: '400',
    color: '#9CA3AF',
    lineHeight: 15,
  },
});
