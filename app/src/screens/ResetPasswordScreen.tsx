import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AuthTheme } from '../constants/authTheme';
import { AuthChrome } from '../components/AuthChrome';
import { AuthPrimaryButton } from '../components/AuthPrimaryButton';
import { AuthTextField } from '../components/AuthTextField';
import { PasswordFieldIcon } from '../components/AuthFieldIcons';
import { resetPassword } from '../services/authApi';

interface ResetPasswordScreenProps {
  initialToken?: string;
  onBack: () => void;
  onDone: () => void;
}

export const ResetPasswordScreen: React.FC<ResetPasswordScreenProps> = ({
  initialToken = '',
  onBack,
  onDone,
}) => {
  const [token, setToken] = useState(initialToken);
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    setError(null);
    setMessage(null);
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setLoading(true);
    try {
      const msg = await resetPassword({
        token,
        password,
        confirmPassword,
      });
      setMessage(msg);
      setTimeout(onDone, 1200);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Reset failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthChrome>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.flex}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          bounces={false}
        >
          <Pressable style={styles.backBtn} onPress={onBack} hitSlop={12}>
            <Ionicons
              name="arrow-undo-outline"
              size={26}
              color={AuthTheme.textPrimary}
            />
          </Pressable>

          <Text style={styles.heading}>Reset Password</Text>
          <Text style={styles.subtitle}>
            Paste the token from your email, then choose a new strong password.
          </Text>

          <AuthTextField
            leftIcon={
              <View style={{ width: 28, marginRight: 10, alignItems: 'center' }}>
                <Ionicons name="key-outline" size={18} color={AuthTheme.accent} />
              </View>
            }
            autoCapitalize="none"
            autoCorrect={false}
            value={token}
            onChangeText={setToken}
            placeholder="Reset Token"
          />
          <AuthTextField
            leftIcon={<PasswordFieldIcon />}
            secure
            textContentType="newPassword"
            value={password}
            onChangeText={setPassword}
            placeholder="New Password"
          />
          <AuthTextField
            leftIcon={<PasswordFieldIcon />}
            secure
            textContentType="newPassword"
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            placeholder="Confirm New Password"
          />

          {error ? <Text style={styles.error}>{error}</Text> : null}
          {message ? <Text style={styles.success}>{message}</Text> : null}

          <AuthPrimaryButton
            label="Update Password"
            onPress={onSubmit}
            loading={loading}
          />
        </ScrollView>
      </KeyboardAvoidingView>
    </AuthChrome>
  );
};

const styles = StyleSheet.create({
  flex: { flex: 1 },
  scroll: {
    flexGrow: 1,
    paddingHorizontal: 28,
    paddingTop: 36,
    paddingBottom: 28,
  },
  backBtn: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    marginBottom: 16,
  },
  heading: {
    color: AuthTheme.textPrimary,
    fontSize: 30,
    fontWeight: '800',
    marginBottom: 10,
  },
  subtitle: {
    color: AuthTheme.textSecondary,
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 28,
  },
  error: {
    color: AuthTheme.error,
    fontSize: 13,
    marginBottom: 10,
  },
  success: {
    color: AuthTheme.accent,
    fontSize: 13,
    marginBottom: 10,
  },
});
