import React from 'react';
import { View, StyleSheet, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AuthTheme } from '../constants/authTheme';

/** Solid dark auth canvas — no decorative glow circles. */
export const AuthChrome: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <View style={styles.root}>
    <StatusBar barStyle="light-content" backgroundColor={AuthTheme.background} />
    <SafeAreaView style={styles.safe}>{children}</SafeAreaView>
  </View>
);

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: AuthTheme.background,
  },
  safe: { flex: 1 },
});
