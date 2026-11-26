import React, { useState } from 'react';
import {
  View,
  TextInput,
  Pressable,
  StyleSheet,
  Platform,
  TextInputProps,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AuthTheme } from '../constants/authTheme';

interface AuthTextFieldProps extends Omit<TextInputProps, 'secureTextEntry'> {
  leftIcon: React.ReactNode;
  secure?: boolean;
}

export const AuthTextField: React.FC<AuthTextFieldProps> = ({
  leftIcon,
  secure = false,
  style,
  ...rest
}) => {
  const [hidden, setHidden] = useState(secure);

  return (
    <View style={styles.field}>
      {leftIcon}
      <TextInput
        {...rest}
        style={[styles.input, style]}
        placeholderTextColor={AuthTheme.textMuted}
        secureTextEntry={secure ? hidden : false}
      />
      {secure ? (
        <Pressable
          onPress={() => setHidden((v) => !v)}
          hitSlop={10}
          accessibilityRole="button"
          accessibilityLabel={hidden ? 'Show password' : 'Hide password'}
        >
          <Ionicons
            name={hidden ? 'eye-outline' : 'eye-off-outline'}
            size={20}
            color={AuthTheme.textMuted}
          />
        </Pressable>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  field: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: AuthTheme.inputBg,
    borderRadius: 999,
    height: 54,
    paddingHorizontal: 18,
    marginBottom: 12,
  },
  input: {
    flex: 1,
    color: AuthTheme.textPrimary,
    fontSize: 15,
    paddingVertical: Platform.OS === 'ios' ? 14 : 10,
  },
});
