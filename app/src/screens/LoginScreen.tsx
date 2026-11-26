import React, { useEffect, useState } from 'react';
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
} from '../components/AuthFieldIcons';
import { useAuth } from '../context/AuthContext';
import {
  getLastGreetingName,
  getRememberedEmail,
} from '../services/authStorage';

interface LoginScreenProps {
  onGoRegister: () => void;
  onForgotPassword: () => void;
  onOpenTerms: () => void;
  onOpenPrivacy: () => void;
}

export const LoginScreen: React.FC<LoginScreenProps> = ({
  onGoRegister,
  onForgotPassword,
  onOpenTerms,
  onOpenPrivacy,
}) => {
  const { signIn } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [greetingName, setGreetingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      const [name, remembered] = await Promise.all([
        getLastGreetingName(),
        getRememberedEmail(),
      ]);
      if (!alive) return;
      setGreetingName(name);
      if (remembered) {
        setEmail(remembered);
        setRememberMe(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const heading = greetingName
    ? `Welcome Back, ${greetingName}!`
    : 'Welcome Back!';

  const onSubmit = async () => {
    setError(null);
    if (!email.trim()) {
      setError('Please enter your email');
      return;
    }
    if (!password) {
      setError('Please enter your password');
      return;
    }
    setLoading(true);
    try {
      await signIn(email.trim(), password, rememberMe);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Sign in failed');
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
          <Pressable style={styles.backBtn} hitSlop={12}>
            <Ionicons
              name="arrow-undo-outline"
              size={26}
              color={AuthTheme.textPrimary}
            />
          </Pressable>

          <Text style={styles.heading}>{heading}</Text>
          <Text style={styles.subtitle}>Your Soft Life Era! Login to Plan.</Text>

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
            textContentType="password"
            value={password}
            onChangeText={setPassword}
            placeholder="Your Password"
          />

          <View style={styles.row}>
            <Pressable
              style={styles.rememberWrap}
              onPress={() => setRememberMe((v) => !v)}
              hitSlop={8}
            >
              <View
                style={[styles.checkbox, rememberMe && styles.checkboxChecked]}
              >
                {rememberMe ? (
                  <Ionicons name="checkmark" size={11} color="#000" />
                ) : null}
              </View>
              <Text style={styles.rememberText}>Remember Me.</Text>
            </Pressable>
            <Pressable onPress={onForgotPassword}>
              <Text style={styles.forgot}>Forgot Password?</Text>
            </Pressable>
          </View>

          {error ? <Text style={styles.error}>{error}</Text> : null}

          <AuthPrimaryButton
            label="Login"
            onPress={onSubmit}
            loading={loading}
          />

          <AuthSocialRow />

          <View style={styles.footer}>
            <Text style={styles.footerText}>Don't have an Account yet? </Text>
            <Pressable onPress={onGoRegister} hitSlop={6}>
              <Text style={styles.footerAccent}>Sign Up</Text>
            </Pressable>
          </View>

          <View style={styles.legalRow}>
            <Pressable onPress={onOpenTerms}>
              <Text style={styles.legalLink}>Terms of Use</Text>
            </Pressable>
            <Text style={styles.legalSep}> · </Text>
            <Pressable onPress={onOpenPrivacy}>
              <Text style={styles.legalLink}>Privacy Policy</Text>
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
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: -0.5,
    lineHeight: 38,
  },
  subtitle: {
    marginTop: 10,
    marginBottom: 32,
    color: AuthTheme.textSecondary,
    fontSize: 15,
    lineHeight: 22,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 18,
  },
  rememberWrap: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkbox: {
    width: 14,
    height: 14,
    marginRight: 8,
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
  rememberText: {
    color: AuthTheme.textPrimary,
    fontSize: 13,
  },
  forgot: {
    color: AuthTheme.accent,
    fontSize: 13,
    fontWeight: '600',
  },
  error: {
    color: AuthTheme.error,
    fontSize: 13,
    marginBottom: 10,
  },
  footer: {
    marginTop: 36,
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
  legalRow: {
    marginTop: 16,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  legalLink: {
    color: AuthTheme.textMuted,
    fontSize: 12,
    textDecorationLine: 'underline',
  },
  legalSep: {
    color: AuthTheme.textMuted,
    fontSize: 12,
  },
});
