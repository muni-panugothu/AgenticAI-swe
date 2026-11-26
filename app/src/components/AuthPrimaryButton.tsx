import React, { createElement } from 'react';
import {
  View,
  Text,
  Pressable,
  ActivityIndicator,
  type ViewStyle,
  type TextStyle,
} from 'react-native';

interface AuthPrimaryButtonProps {
  label: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
}

/**
 * Lime CTA.
 * createElement + inline styles bypass NativeWind JSX remapping that was
 * stripping flex centering and pinning "Login" / "Sign Up" to the top.
 */
export const AuthPrimaryButton: React.FC<AuthPrimaryButtonProps> = ({
  label,
  onPress,
  loading = false,
  disabled = false,
}) => {
  const inactive = loading || disabled;

  const shellStyle: ViewStyle = {
    width: '100%',
    height: 54,
    marginTop: 10,
    marginBottom: 4,
    borderRadius: 999,
    backgroundColor: '#D4FF00',
    alignSelf: 'stretch',
    overflow: 'hidden',
    opacity: inactive ? 0.55 : 1,
  };

  const rowStyle: ViewStyle = {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const labelStyle: TextStyle = {
    color: '#000000',
    fontSize: 17,
    fontWeight: '800',
    textAlign: 'center',
    includeFontPadding: false,
    textAlignVertical: 'center',
    // Single-line optical vertical center (Android-safe)
    lineHeight: 54,
    height: 54,
  };

  const content = loading
    ? createElement(ActivityIndicator, { color: '#000000' })
    : createElement(Text, { style: labelStyle, numberOfLines: 1 }, label);

  return createElement(
    View,
    { style: shellStyle },
    createElement(View, { style: rowStyle, pointerEvents: 'box-none' }, content),
    createElement(Pressable, {
      accessibilityRole: 'button' as const,
      accessibilityLabel: label,
      disabled: inactive,
      onPress,
      style: {
        position: 'absolute',
        top: 0,
        right: 0,
        bottom: 0,
        left: 0,
      },
    })
  );
};
