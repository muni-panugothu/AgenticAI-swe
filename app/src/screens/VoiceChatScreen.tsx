import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, StatusBar, useWindowDimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { Header } from '../components/Header';
import { AIAnimation } from '../components/AIAnimation';
import { VoiceControls } from '../components/VoiceControls';
import { Colors } from '../constants/colors';
import type { RootStackParamList } from '../types';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

export const VoiceChatScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const [isListening, setIsListening] = useState(true);
  const { width } = useWindowDimensions();
  const animationSize = Math.min(Math.max(width * 0.62, 220), 320);

  const handleBack = useCallback(() => {
    navigation.goBack();
  }, [navigation]);

  const handleVoiceToggle = useCallback(() => {
    setIsListening((prev) => !prev);
  }, []);

  const handleChat = useCallback(() => {
    navigation.navigate('SmartChat');
  }, [navigation]);

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor="transparent" translucent />
      <LinearGradient
        colors={['#FBF9FF', '#F0EAFF', '#F5F0FF']}
        style={StyleSheet.absoluteFill}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
      />

      <SafeAreaView style={styles.container} edges={['top']}>
        <Header
          title="Voice Chat"
          onBack={handleBack}
          rightIcon="play-circle-outline"
        />

        <View style={styles.content}>
          <Text style={styles.listeningText}>I'm Listening You...</Text>

          <View style={styles.animationContainer}>
            <AIAnimation size={animationSize} autoPlay={isListening} loop />
          </View>

          <Text style={styles.description}>
            Hello 👋 I'm here to help you learn,{'\n'}explore new things and tackle{'\n'}tricky questions
          </Text>
        </View>

        <VoiceControls
          onChat={handleChat}
          onVoice={handleVoiceToggle}
          onClose={handleBack}
          isListening={isListening}
        />
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 36,
  },
  listeningText: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text.primary,
    textAlign: 'center',
    marginBottom: 28,
    letterSpacing: -0.3,
  },
  animationContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 28,
  },
  description: {
    fontSize: 14,
    fontWeight: '400',
    color: Colors.text.secondary,
    textAlign: 'center',
    lineHeight: 22,
  },
});
