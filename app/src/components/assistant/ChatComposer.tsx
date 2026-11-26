import React, { createElement, useEffect, useState } from 'react';
import {
  View,
  TextInput,
  Pressable,
  ActivityIndicator,
  Platform,
  Keyboard,
  StyleSheet,
  type ViewStyle,
  type TextStyle,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { AssistantTheme } from '../../constants/assistantTheme';
import { VoiceWaveform } from './VoiceWaveform';

interface ChatComposerProps {
  placeholder?: string;
  onSend?: (text: string) => void;
  onAttachmentPress?: () => void;
  /** Advanced voice chat (green pulse). Mic uses inline dictation UI. */
  onVoicePress?: () => void;
  /** Stop in-flight SSE generation (ChatGPT-style). */
  onStop?: () => void;
  disabled?: boolean;
  uploading?: boolean;
  /** True while the assistant reply is streaming. */
  streaming?: boolean;
  showAttachment?: boolean;
  showVoice?: boolean;
}

type ComposerMode = 'text' | 'recording' | 'paused';

/**
 * ChatGPT-style composer.
 * Mic → inline voice bar: X | waveform | stop | send
 * Uses createElement so NativeWind cannot strip layout styles.
 */
export const ChatComposer: React.FC<ChatComposerProps> = ({
  placeholder = 'Ask anything',
  onSend,
  onAttachmentPress,
  onVoicePress,
  onStop,
  disabled = false,
  uploading = false,
  streaming = false,
  showAttachment = true,
  showVoice = true,
}) => {
  const insets = useSafeAreaInsets();
  const [text, setText] = useState('');
  const [mode, setMode] = useState<ComposerMode>('text');
  const [keyboardLift, setKeyboardLift] = useState(0);
  const hasText = text.trim().length > 0;
  const inVoice = mode === 'recording' || mode === 'paused';

  useEffect(() => {
    if (Platform.OS !== 'android') return;

    const onShow = Keyboard.addListener('keyboardDidShow', (event) => {
      const height = event.endCoordinates?.height ?? 0;
      setKeyboardLift(Math.max(0, height - insets.bottom));
    });
    const onHide = Keyboard.addListener('keyboardDidHide', () => {
      setKeyboardLift(0);
    });

    return () => {
      onShow.remove();
      onHide.remove();
    };
  }, [insets.bottom]);

  const handleSendText = () => {
    if (disabled || uploading || !hasText) return;
    onSend?.(text.trim());
    setText('');
  };

  const startVoice = () => {
    if (disabled || uploading) return;
    Keyboard.dismiss();
    setMode('recording');
  };

  const cancelVoice = () => {
    setMode('text');
  };

  const toggleStop = () => {
    setMode((prev) => (prev === 'recording' ? 'paused' : 'recording'));
  };

  const sendVoice = () => {
    // UI-only capture for now — send typed draft if present, else a short marker.
    const payload = text.trim() || '🎤 Voice message';
    onSend?.(payload);
    setText('');
    setMode('text');
  };

  const handlePrimary = () => {
    if (hasText) {
      handleSendText();
      return;
    }
    // Green pulse → advanced voice chat screen when provided
    if (onVoicePress) {
      onVoicePress();
      return;
    }
    startVoice();
  };

  const outerStyle: ViewStyle = {
    paddingHorizontal: 12,
    paddingBottom: Platform.OS === 'ios' ? 8 : 12,
    paddingTop: 4,
    marginBottom: keyboardLift > 0 && !inVoice ? keyboardLift : 0,
  };

  const shellStyle: ViewStyle = {
    position: 'relative',
    minHeight: 56,
    borderRadius: 28,
    backgroundColor: AssistantTheme.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: AssistantTheme.border,
    justifyContent: 'center',
    overflow: 'hidden',
  };

  const voiceRowStyle: ViewStyle = {
    minHeight: 56,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 8,
  };

  const roundBtnStyle: ViewStyle = {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#EFEFEE',
  };

  const sendBlueStyle: ViewStyle = {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0B84FF',
    marginLeft: 6,
  };

  const inputStyle: TextStyle = {
    minHeight: 44,
    maxHeight: 100,
    fontSize: 16,
    lineHeight: 22,
    color: AssistantTheme.textPrimary,
    paddingLeft: showAttachment ? 48 : 16,
    paddingRight: showVoice ? 96 : 56,
    paddingVertical: Platform.OS === 'ios' ? 12 : 10,
    margin: 0,
  };

  const leftDockStyle: ViewStyle = {
    position: 'absolute',
    left: 4,
    top: 0,
    bottom: 0,
    width: 44,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  };

  const rightDockStyle: ViewStyle = {
    position: 'absolute',
    right: 6,
    top: 0,
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 2,
  };

  const hitStyle: ViewStyle = {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  };

  const primaryCircleStyle: ViewStyle = {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 2,
    backgroundColor: hasText ? '#111111' : '#10A37F',
    opacity: disabled || uploading ? 0.45 : 1,
  };

  if (inVoice) {
    const voiceShell: ViewStyle = {
      ...shellStyle,
      ...voiceRowStyle,
      overflow: 'visible',
    };

    return createElement(
      View,
      { style: outerStyle },
      createElement(
        View,
        { style: voiceShell },
        createElement(
          Pressable,
          {
            onPress: cancelVoice,
            accessibilityLabel: 'Cancel voice',
          },
          createElement(
            View,
            { style: roundBtnStyle },
            createElement(Ionicons, {
              name: 'close',
              size: 22,
              color: '#111111',
            })
          )
        ),
        createElement(VoiceWaveform, { active: mode === 'recording' }),
        createElement(
          Pressable,
          {
            onPress: toggleStop,
            accessibilityLabel: mode === 'recording' ? 'Stop' : 'Resume',
          },
          createElement(
            View,
            { style: roundBtnStyle },
            createElement(Ionicons, {
              name: mode === 'recording' ? 'square' : 'play',
              size: mode === 'recording' ? 14 : 18,
              color: '#111111',
            })
          )
        ),
        createElement(
          Pressable,
          {
            onPress: sendVoice,
            accessibilityLabel: 'Send voice',
          },
          createElement(
            View,
            { style: sendBlueStyle },
            createElement(Ionicons, {
              name: 'arrow-up',
              size: 20,
              color: '#FFFFFF',
            })
          )
        )
      )
    );
  }

  return createElement(
    View,
    { style: outerStyle },
    createElement(
      View,
      { style: shellStyle },
      showAttachment
        ? createElement(
            View,
            { style: leftDockStyle },
            createElement(
              Pressable,
              {
                onPress: onAttachmentPress,
                disabled: disabled || uploading,
                hitSlop: 6,
                accessibilityLabel: 'Attach',
              },
              createElement(
                View,
                { style: hitStyle },
                uploading
                  ? createElement(ActivityIndicator, {
                      size: 'small',
                      color: AssistantTheme.icon,
                    })
                  : createElement(Ionicons, {
                      name: 'add',
                      size: 26,
                      color: '#111111',
                    })
              )
            )
          )
        : null,
      createElement(TextInput, {
        style: inputStyle,
        placeholder,
        placeholderTextColor: AssistantTheme.textMuted,
        value: text,
        onChangeText: setText,
        editable: !disabled && !uploading,
        multiline: true,
        maxLength: 4000,
        textAlignVertical: 'center',
        returnKeyType: 'send',
        blurOnSubmit: false,
        onSubmitEditing: handleSendText,
      }),
      createElement(
        View,
        { style: rightDockStyle },
        showVoice
          ? createElement(
              Pressable,
              {
                onPress: startVoice,
                hitSlop: 6,
                accessibilityLabel: 'Voice input',
              },
              createElement(
                View,
                { style: hitStyle },
                createElement(Ionicons, {
                  name: 'mic-outline',
                  size: 22,
                  color: '#111111',
                })
              )
            )
          : null,
        streaming && onStop
          ? createElement(
              Pressable,
              {
                onPress: onStop,
                hitSlop: 4,
                accessibilityLabel: 'Stop generating',
              },
              createElement(
                View,
                {
                  style: {
                    ...primaryCircleStyle,
                    backgroundColor: '#111111',
                  },
                },
                createElement(Ionicons, {
                  name: 'stop',
                  size: 16,
                  color: '#FFFFFF',
                })
              )
            )
          : createElement(
              Pressable,
              {
                onPress: handlePrimary,
                disabled:
                  disabled ||
                  uploading ||
                  (!hasText && !onVoicePress && !showVoice),
                hitSlop: 4,
                accessibilityLabel: hasText
                  ? 'Send message'
                  : 'Start voice chat',
              },
              createElement(
                View,
                { style: primaryCircleStyle },
                createElement(Ionicons, {
                  name: hasText ? 'arrow-up' : 'pulse',
                  size: hasText ? 20 : 18,
                  color: '#FFFFFF',
                })
              )
            )
      )
    )
  );
};
