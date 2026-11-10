import React from 'react';
import { View, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors } from '../constants/colors';

interface VoiceControlsProps {
  onChat?: () => void;
  onVoice?: () => void;
  onClose?: () => void;
  isListening?: boolean;
}

export const VoiceControls: React.FC<VoiceControlsProps> = ({
  onChat,
  onVoice,
  onClose,
  isListening = false,
}) => {
  const insets = useSafeAreaInsets();

  return (
    <View style={[styles.container, { paddingBottom: Math.max(insets.bottom, 20) }]}>
      <Pressable
        onPress={onChat}
        style={({ pressed }) => [styles.sideButton, pressed && styles.pressed]}
      >
        <Ionicons name="chatbubble-ellipses-outline" size={22} color={Colors.text.secondary} />
      </Pressable>

      <Pressable
        onPress={onVoice}
        style={({ pressed }) => [styles.centerButtonWrap, pressed && styles.centerPressed]}
      >
        <View style={styles.centerGlow} />
        <LinearGradient
          colors={[Colors.gradient.purpleStart, Colors.gradient.purpleEnd]}
          style={styles.centerButton}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <Ionicons
            name={isListening ? 'mic' : 'mic-outline'}
            size={28}
            color="#FFFFFF"
          />
        </LinearGradient>
      </Pressable>

      <Pressable
        onPress={onClose}
        style={({ pressed }) => [styles.sideButton, pressed && styles.pressed]}
      >
        <Ionicons name="close" size={22} color={Colors.text.secondary} />
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 32,
    paddingTop: 20,
  },
  sideButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: Colors.border.light,
    shadowColor: Colors.shadow.light,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 6,
    elevation: 3,
  },
  pressed: {
    opacity: 0.7,
    transform: [{ scale: 0.95 }],
  },
  centerButtonWrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  centerButton: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: Colors.primary[600],
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 8,
  },
  centerPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.95 }],
  },
  centerGlow: {
    position: 'absolute',
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: 'rgba(139, 92, 246, 0.12)',
  },
});
