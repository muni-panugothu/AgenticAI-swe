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
import { EmailFieldIcon } from '../components/AuthFieldIcons';
import { requestPasswordReset } from '../services/authApi';

interface ForgotPasswordScreenProps {
  onBack: () => void;
  onHaveToken: () => void;
}

export const ForgotPasswordScreen: React.FC<ForgotPasswordScreenProps> = ({
  onBack,
  onHaveToken,
}) => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    setError(null);
    setMessage(null);
    setLoading(true);
    try {
      const msg = await requestPasswordReset(email.trim());
      setMessage(msg);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed');
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

          <Text style={styles.heading}>Forgot Password?</Text>
          <Text style={styles.subtitle}>
            Enter your email and we'll send a secure one-time reset link.
          </Text>

          <AuthTextField
            leftIcon={<EmailFieldIcon />}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            textContentType="emailAddress"
            value={email}
            onChangeText={setEmail}
            placeholder="Your Email"
          />

          {error ? <Text style={styles.error}>{error}</Text> : null}
          {message ? <Text style={styles.success}>{message}</Text> : null}

          <AuthPrimaryButton
            label="Send Reset Link"
            onPress={onSubmit}
            loading={loading}
          />

          <Pressable style={styles.linkWrap} onPress={onHaveToken}>
            <Text style={styles.link}>Already have a reset token?</Text>
          </Pressable>
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
    lineHeight: 18,
    marginBottom: 10,
  },
  linkWrap: {
    marginTop: 24,
    alignItems: 'center',
  },
  link: {
    color: AuthTheme.accent,
    fontSize: 14,
    fontWeight: '600',
  },
});
