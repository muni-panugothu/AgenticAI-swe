import React, { useState } from 'react';
import {
  View,
  TextInput,
  Pressable,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';

interface AIInputProps {
  placeholder?: string;
  onSend?: (text: string) => void;
  onVoicePress?: () => void;
  onAttachmentPress?: () => void;
  showAttachment?: boolean;
  showVoice?: boolean;
  disabled?: boolean;
  uploading?: boolean;
}

export const AIInput: React.FC<AIInputProps> = ({
  placeholder = 'Ask me anything...',
  onSend,
  onVoicePress,
  onAttachmentPress,
  showAttachment = false,
  showVoice = false,
  disabled = false,
  uploading = false,
}) => {
  const [text, setText] = useState('');
  const hasText = text.trim().length > 0;
  const showSend = hasText;

  const handleSend = () => {
    if (disabled || uploading) return;
    const trimmed = text.trim();
    if (!trimmed) return;
    onSend?.(trimmed);
    setText('');
  };

  const handleActionPress = () => {
    if (disabled || uploading) return;
    if (showSend) {
      handleSend();
      return;
    }
    if (showVoice) {
      onVoicePress?.();
    }
  };

  return (
    <View style={styles.container}>
      {showAttachment && (
        <Pressable
          onPress={onAttachmentPress}
          disabled={disabled || uploading}
          accessibilityRole="button"
          accessibilityLabel="Upload document"
          style={({ pressed }) => [
            styles.attachButton,
            (pressed || uploading) && styles.pressed,
            (disabled || uploading) && styles.disabled,
          ]}
        >
          {uploading ? (
            <ActivityIndicator size="small" color="#7C3AED" />
          ) : (
            <Ionicons name="add" size={24} color="#6B7280" />
          )}
        </Pressable>
      )}

      <View style={styles.inputWrapper}>
        {!showAttachment && (
          <LinearGradient
            colors={['#C4B5FD', '#8B5CF6', '#A78BFA']}
            style={styles.aiIcon}
            start={{ x: 0.1, y: 0 }}
            end={{ x: 0.9, y: 1 }}
          >
            <View style={styles.aiIconInner} />
          </LinearGradient>
        )}
        <TextInput
          style={styles.input}
          placeholder={placeholder}
          placeholderTextColor="#A1A1AA"
          value={text}
          onChangeText={setText}
          onSubmitEditing={handleSend}
          returnKeyType="send"
          editable={!disabled && !uploading}
          multiline
          maxLength={4000}
        />
      </View>

      {/* Send / mic — outside the input pill (WhatsApp-style) */}
      <Pressable
        onPress={handleActionPress}
        disabled={disabled || uploading || (!showSend && !showVoice)}
        accessibilityRole="button"
        accessibilityLabel={showSend ? 'Send message' : 'Voice chat'}
        style={({ pressed }) => [pressed && styles.pressed]}
      >
        <LinearGradient
          colors={['#9B7DFF', '#7C3AED']}
          style={[
            styles.actionButton,
            (!showSend && !showVoice) && styles.actionButtonDisabled,
          ]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
        >
          <Ionicons
            name={showSend ? 'send' : 'mic'}
            size={showSend ? 20 : 22}
            color="#FFFFFF"
            style={showSend ? styles.sendIcon : undefined}
          />
        </LinearGradient>
      </Pressable>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 16,
    gap: 8,
  },
  attachButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 2,
  },
  inputWrapper: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderRadius: 24,
    paddingLeft: 14,
    paddingRight: 14,
    minHeight: 48,
    maxHeight: 120,
    paddingVertical: 10,
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 4,
  },
  aiIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  aiIconInner: {
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: 'rgba(255,255,255,0.85)',
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: '#1F1F2E',
    paddingVertical: 0,
    maxHeight: 100,
  },
  actionButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 0,
  },
  actionButtonDisabled: {
    opacity: 0.5,
  },
  sendIcon: {
    marginLeft: 2,
    marginTop: 1,
  },
  pressed: {
    opacity: 0.85,
    transform: [{ scale: 0.96 }],
  },
  disabled: {
    opacity: 0.5,
  },
});
