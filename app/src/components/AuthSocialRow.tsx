import React from 'react';
import { View, Text, Pressable, StyleSheet, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AuthTheme } from '../constants/authTheme';
import { GoogleMark } from './GoogleMark';

/** Visual social row — lime border matches primary CTA; icons centered. */
export const AuthSocialRow: React.FC = () => {
  const onPress = (provider: 'Google' | 'Apple') => {
    Alert.alert(
      `${provider} sign-in unavailable`,
      `${provider} OAuth is not configured for this build. Please use email authentication.`
    );
  };

  return (
    <>
      <View style={styles.orRow}>
        <View style={styles.orLine} />
        <Text style={styles.orText}>or</Text>
        <View style={styles.orLine} />
      </View>
      <View style={styles.socialRow}>
        <View style={styles.socialShell}>
          <View style={styles.iconCenter} pointerEvents="none">
            <GoogleMark size={22} />
          </View>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => onPress('Google')}
            accessibilityRole="button"
            accessibilityLabel="Continue with Google"
          />
        </View>
        <View style={styles.socialShell}>
          <View style={styles.iconCenter} pointerEvents="none">
            <Ionicons
              name="logo-apple"
              size={24}
              color={AuthTheme.textPrimary}
            />
          </View>
          <Pressable
            style={StyleSheet.absoluteFillObject}
            onPress={() => onPress('Apple')}
            accessibilityRole="button"
            accessibilityLabel="Continue with Apple"
          />
        </View>
      </View>
    </>
  );
};

const styles = StyleSheet.create({
  orRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 22,
    marginBottom: 20,
  },
  orLine: {
    flex: 1,
    height: StyleSheet.hairlineWidth,
    backgroundColor: AuthTheme.divider,
  },
  orText: {
    marginHorizontal: 14,
    color: AuthTheme.textMuted,
    fontSize: 13,
    textTransform: 'lowercase',
  },
  socialRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  socialShell: {
    width: 52,
    height: 52,
    marginHorizontal: 11,
    borderRadius: 26,
    borderWidth: 1.5,
    borderColor: '#D4FF00',
    backgroundColor: '#000000',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconCenter: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
