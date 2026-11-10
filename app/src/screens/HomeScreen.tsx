import React, { useCallback } from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  StatusBar,
  useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { HomeHeader } from '../components/HomeHeader';
import { GreetingSection } from '../components/GreetingSection';
import { AIInput } from '../components/AIInput';
import { QuickActionGrid } from '../components/QuickActionGrid';
import { AIAnimation } from '../components/AIAnimation';
import { BottomNavigation } from '../components/BottomNavigation';
import type { RootStackParamList } from '../types';

type NavigationProp = NativeStackNavigationProp<RootStackParamList>;

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const { width, height } = useWindowDimensions();
  const animationSize = Math.min(
    Math.max(width * 0.6, 210),
    Math.min(height * 0.3, 290)
  );

  const handleTabPress = useCallback(
    (tabId: string) => {
      if (tabId === 'ai') {
        navigation.navigate('VoiceChat');
      } else if (tabId === 'history') {
        navigation.navigate('SmartChat');
      }
    },
    [navigation]
  );

  const handleSend = useCallback(
    (text: string) => {
      navigation.navigate('SmartChat', { initialMessage: text });
    },
    [navigation]
  );

  const handleActionPress = useCallback(
    (actionId: string) => {
      navigation.navigate('SmartChat', { actionId });
    },
    [navigation]
  );

  return (
    <View style={styles.root}>
      <StatusBar barStyle="dark-content" backgroundColor="#F7F7FD" />
      <LinearGradient
        colors={['#FAFAFE', '#F7F7FD', '#F0EBFF']}
        style={StyleSheet.absoluteFill}
        start={{ x: 0.5, y: 0 }}
        end={{ x: 0.5, y: 1 }}
      />

      <SafeAreaView style={styles.container} edges={['top']}>
        <HomeHeader
          onProfilePress={() => navigation.navigate('Profile')}
        />

        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          bounces
        >
          <GreetingSection />

          <View style={styles.inputSection}>
            <AIInput onSend={handleSend} />
          </View>

          <QuickActionGrid onActionPress={handleActionPress} />

          <View style={styles.animationSection}>
            <AIAnimation size={animationSize} autoPlay loop />
          </View>
        </ScrollView>
      </SafeAreaView>

      <BottomNavigation activeTab="home" onTabPress={handleTabPress} />
    </View>
  );
};

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#F7F7FD',
  },
  container: {
    flex: 1,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
    flexGrow: 1,
  },
  inputSection: {
    marginTop: 20,
    marginBottom: 20,
  },
  animationSection: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
    marginBottom: 8,
    flexGrow: 1,
    minHeight: 190,
  },
});
