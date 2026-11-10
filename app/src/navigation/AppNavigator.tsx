import React, { useState } from 'react';
import { ActivityIndicator, View, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { AuthProvider, useAuth } from '../context/AuthContext';
import { HomeScreen } from '../screens/HomeScreen';
import { VoiceChatScreen } from '../screens/VoiceChatScreen';
import { SmartChatScreen } from '../screens/SmartChatScreen';
import { LoginScreen } from '../screens/LoginScreen';
import { RegisterScreen } from '../screens/RegisterScreen';
import { ProfileScreen } from '../screens/ProfileScreen';
import { Colors } from '../constants/colors';
import type { RootStackParamList } from '../types';

const Stack = createNativeStackNavigator<RootStackParamList>();

const AppStack: React.FC = () => (
  <Stack.Navigator
    initialRouteName="MainTabs"
    screenOptions={{
      headerShown: false,
      animation: 'slide_from_right',
      contentStyle: { backgroundColor: 'transparent' },
    }}
  >
    <Stack.Screen name="MainTabs" component={HomeScreen} />
    <Stack.Screen
      name="VoiceChat"
      component={VoiceChatScreen}
      options={{ animation: 'fade_from_bottom' }}
    />
    <Stack.Screen name="SmartChat" component={SmartChatScreen} />
    <Stack.Screen name="Profile" component={ProfileScreen} />
  </Stack.Navigator>
);

const AuthStack: React.FC = () => {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  return mode === 'login' ? (
    <LoginScreen onGoRegister={() => setMode('register')} />
  ) : (
    <RegisterScreen onGoLogin={() => setMode('login')} />
  );
};

const RootNavigation: React.FC = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color={Colors.primary[600]} />
      </View>
    );
  }

  return (
    <NavigationContainer>
      {user ? <AppStack /> : <AuthStack />}
    </NavigationContainer>
  );
};

export const AppNavigator: React.FC = () => (
  <AuthProvider>
    <RootNavigation />
  </AuthProvider>
);

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FAFAFE',
  },
});
