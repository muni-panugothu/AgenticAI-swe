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
import { AuthSocialRow } from '../components/AuthSocialRow';
import { AuthTextField } from '../components/AuthTextField';
import {
  EmailFieldIcon,
  PasswordFieldIcon,
  PersonFieldIcon,
} from '../components/AuthFieldIcons';
import { useAuth } from '../context/AuthContext';

interface RegisterScreenProps {
  onGoLogin: () => void;
  onOpenTerms: () => void;
  onOpenPrivacy: () => void;
}

export const RegisterScreen: React.FC<RegisterScreenProps> = ({
  onGoLogin,
  onOpenTerms,
  onOpenPrivacy,
}) => {
  const { signUp } = useAuth();
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [acceptedLegal, setAcceptedLegal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    setError(null);
    if (!displayName.trim()) {
      setError('Please enter your name');
      return;
    }
    if (!email.trim()) {
      setError('Please enter your email');
      return;
    }
    if (!password) {
      setError('Please enter a password');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    if (!/[A-Z]/.test(password) || !/[a-z]/.test(password) || !/\d/.test(password)) {
      setError('Password needs upper, lower, and a number');
      return;
    }
    if (!acceptedLegal) {
      setError('Please accept the Terms of Use & Privacy Policy');
      return;
    }

    setLoading(true);
    try {
      await signUp({
        displayName: displayName.trim(),
        email: email.trim(),
        password,
        acceptedTerms: true,
        acceptedPrivacy: true,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registration failed');
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
          showsVerticalScrollIndicator={false}
        >
          <Pressable style={styles.backBtn} hitSlop={12} onPress={onGoLogin}>
            <Ionicons
              name="arrow-undo-outline"
              size={26}
              color={AuthTheme.textPrimary}
            />
          </Pressable>

          <Text style={styles.heading}>Create Your Account</Text>
          <Text style={styles.subtitle}>
            Cancel Being Broke! Plan every Penny!
          </Text>

          <AuthTextField
            leftIcon={<PersonFieldIcon />}
            autoCapitalize="words"
            textContentType="name"
            value={displayName}
            onChangeText={setDisplayName}
            placeholder="Your Name"
          />
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
          <AuthTextField
            leftIcon={<PasswordFieldIcon />}
            secure
            textContentType="newPassword"
            value={password}
            onChangeText={setPassword}
            placeholder="Your Password"
          />

          <Pressable
            style={styles.termsRow}
            onPress={() => setAcceptedLegal((v) => !v)}
          >
            <View
              style={[
                styles.checkbox,
                acceptedLegal && styles.checkboxChecked,
              ]}
            >
              {acceptedLegal ? (
                <Ionicons name="checkmark" size={11} color="#000" />
              ) : null}
            </View>
            <Text style={styles.termsText}>
              I accept the{' '}
              <Text style={styles.termsLink} onPress={onOpenTerms}>
                Terms of Use
              </Text>
              {' & '}
              <Text style={styles.termsLink} onPress={onOpenPrivacy}>
                Privacy Policy
              </Text>
            </Text>
          </Pressable>

          {error ? <Text style={styles.error}>{error}</Text> : null}

          <AuthPrimaryButton
            label="Sign Up"
            onPress={onSubmit}
            loading={loading}
          />

          <AuthSocialRow />

          <View style={styles.footer}>
            <Text style={styles.footerText}>Already have an Account? </Text>
            <Pressable onPress={onGoLogin} hitSlop={6}>
              <Text style={styles.footerAccent}>Sign In</Text>
            </Pressable>
          </View>
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
    paddingTop: 56,
    paddingBottom: 36,
  },
  backBtn: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    marginBottom: 18,
  },
  heading: {
    color: AuthTheme.textPrimary,
    fontSize: 30,
    fontWeight: '800',
    letterSpacing: -0.4,
    lineHeight: 36,
  },
  subtitle: {
    marginTop: 8,
    marginBottom: 28,
    color: AuthTheme.textSecondary,
    fontSize: 14,
    lineHeight: 20,
  },
  termsRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 4,
    marginBottom: 14,
  },
  checkbox: {
    width: 15,
    height: 15,
    marginTop: 2,
    marginRight: 10,
    borderRadius: 3,
    borderWidth: 1.5,
    borderColor: AuthTheme.checkboxBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxChecked: {
    backgroundColor: AuthTheme.accent,
    borderColor: AuthTheme.accent,
  },
  termsText: {
    flex: 1,
    color: AuthTheme.textPrimary,
    fontSize: 13,
    lineHeight: 18,
  },
  termsLink: {
    color: AuthTheme.accent,
    fontWeight: '700',
  },
  error: {
    color: AuthTheme.error,
    fontSize: 13,
    marginBottom: 10,
  },
  footer: {
    marginTop: 32,
    flexDirection: 'row',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  footerText: {
    color: AuthTheme.textPrimary,
    fontSize: 14,
  },
  footerAccent: {
    color: AuthTheme.accent,
    fontSize: 14,
    fontWeight: '800',
  },
});
