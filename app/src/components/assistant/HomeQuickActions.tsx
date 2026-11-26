import React, { createElement } from 'react';
import { View, Text, Pressable, type ViewStyle, type TextStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export type HomeQuickActionId = 'voice' | 'image' | 'write';

interface HomeQuickActionsProps {
  onAction: (id: HomeQuickActionId) => void;
}

const ACTIONS: Array<{
  id: HomeQuickActionId;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
}> = [
  { id: 'voice', label: 'Start a voice chat', icon: 'headset-outline' },
  {
    id: 'image',
    label: 'Create an image or sticker',
    icon: 'image-outline',
  },
  { id: 'write', label: 'Write or edit', icon: 'create-outline' },
];

const wrapStyle: ViewStyle = {
  width: '100%',
  paddingHorizontal: 28,
  paddingBottom: 20,
};

const rowStyle: ViewStyle = {
  flexDirection: 'row',
  alignItems: 'center',
  width: '100%',
  paddingVertical: 14,
};

const iconSlotStyle: ViewStyle = {
  width: 28,
  height: 28,
  marginRight: 14,
  alignItems: 'center',
  justifyContent: 'center',
};

const labelStyle: TextStyle = {
  flexShrink: 1,
  fontSize: 16,
  lineHeight: 22,
  color: '#5F5F5F',
  fontWeight: '400',
};

export const HomeQuickActions: React.FC<HomeQuickActionsProps> = ({
  onAction,
}) => {
  return createElement(
    View,
    { style: wrapStyle },
    ...ACTIONS.map((action) =>
      createElement(
        Pressable,
        {
          key: action.id,
          onPress: () => onAction(action.id),
          accessibilityRole: 'button',
          accessibilityLabel: action.label,
        },
        createElement(
          View,
          { style: rowStyle },
          createElement(
            View,
            { style: iconSlotStyle },
            createElement(Ionicons, {
              name: action.icon,
              size: 22,
              color: '#5F5F5F',
            })
          ),
          createElement(
            Text,
            { style: labelStyle, numberOfLines: 1 },
            action.label
          )
        )
      )
    )
  );
};
