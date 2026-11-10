import React from 'react';
import { View, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '../constants/colors';

interface MessageActionsProps {
  onCopy?: () => void;
  onLike?: () => void;
  onDislike?: () => void;
  onSpeak?: () => void;
}

export const MessageActions: React.FC<MessageActionsProps> = ({
  onCopy,
  onLike,
  onDislike,
  onSpeak,
}) => {
  const actions = [
    { icon: 'copy-outline', onPress: onCopy },
    { icon: 'thumbs-up-outline', onPress: onLike },
    { icon: 'thumbs-down-outline', onPress: onDislike },
    { icon: 'volume-medium-outline', onPress: onSpeak },
  ];

  return (
    <View style={styles.container}>
      {actions.map((action, index) => (
        <Pressable
          key={index}
          onPress={action.onPress}
          style={({ pressed }) => [styles.button, pressed && styles.pressed]}
        >
          <Ionicons
            name={action.icon as any}
            size={16}
            color={Colors.text.muted}
          />
        </Pressable>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  button: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pressed: {
    opacity: 0.5,
    backgroundColor: Colors.primary[50],
  },
});
