import React from 'react';
import { StyleSheet } from 'react-native';
import { View } from './rn';
import { QuickActionCard } from './QuickActionCard';
import { quickActions, type QuickAction } from '../constants/actions';

interface QuickActionGridProps {
  onActionPress?: (actionId: string) => void;
}

export function QuickActionGrid({ onActionPress }: QuickActionGridProps) {
  const rows: [QuickAction, QuickAction][] = [
    [quickActions[0], quickActions[1]],
    [quickActions[2], quickActions[3]],
  ];

  return (
    <View style={styles.container}>
      {rows.map((pair, rowIndex) => (
        <View key={rowIndex} style={styles.row}>
          {pair.map((action) => (
            <QuickActionCard
              key={action.id}
              action={action}
              onPress={() => onActionPress?.(action.id)}
            />
          ))}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    gap: 12,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
  },
});
