import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export const GreetingSection: React.FC = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.greeting}>Hey, Alex 👋</Text>
      <Text style={styles.heading}>
        How can I help you{'\n'}today?
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    paddingTop: 2,
    paddingBottom: 2,
  },
  greeting: {
    fontSize: 16,
    fontWeight: '500',
    color: '#6B7280',
    marginBottom: 8,
  },
  heading: {
    fontSize: 30,
    fontWeight: '700',
    color: '#111827',
    lineHeight: 36,
    letterSpacing: -0.6,
  },
});
