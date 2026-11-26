import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { BrandMark } from './BrandMark';
import {
  AssistantTheme,
  greetingForNow,
} from '../../constants/assistantTheme';
import { useAuth } from '../../context/AuthContext';

export const HomeGreeting: React.FC = () => {
  const { user } = useAuth();
  const name =
    user?.display_name?.split(/\s+/)[0] ||
    user?.email?.split('@')[0] ||
    'there';
  const capitalized =
    name.charAt(0).toUpperCase() + name.slice(1);

  return (
    <View style={styles.wrap}>
      <BrandMark size={78} />
      <Text style={styles.greeting}>{greetingForNow(capitalized)}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingBottom: 24,
  },
  greeting: {
    marginTop: 28,
    fontSize: 28,
    lineHeight: 36,
    fontWeight: '600',
    color: AssistantTheme.textPrimary,
    textAlign: 'center',
    letterSpacing: -0.4,
    fontFamily: Platform.select({
      ios: 'Georgia',
      android: 'serif',
      default: 'serif',
    }),
  },
});

