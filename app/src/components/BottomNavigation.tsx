import React from 'react';
import { View, Pressable, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BlurView } from 'expo-blur';
import { Platform } from 'react-native';

interface NavItem {
  id: string;
  icon: keyof typeof Ionicons.glyphMap;
  activeIcon: keyof typeof Ionicons.glyphMap;
  isCenter?: boolean;
}

const navItems: NavItem[] = [
  { id: 'home', icon: 'home-outline', activeIcon: 'home' },
  { id: 'history', icon: 'chatbubble-outline', activeIcon: 'chatbubble' },
  { id: 'ai', icon: 'sparkles', activeIcon: 'sparkles', isCenter: true },
  { id: 'files', icon: 'folder-outline', activeIcon: 'folder' },
  { id: 'profile', icon: 'person-outline', activeIcon: 'person' },
];

interface BottomNavigationProps {
  activeTab?: string;
  onTabPress?: (tabId: string) => void;
}

export const BottomNavigation: React.FC<BottomNavigationProps> = ({
  activeTab = 'home',
  onTabPress,
}) => {
  const insets = useSafeAreaInsets();

  const content = (
    <>
      {navItems.map((item) => {
        const isActive = activeTab === item.id;

        if (item.isCenter) {
          return (
            <Pressable
              key={item.id}
              onPress={() => onTabPress?.(item.id)}
              style={({ pressed }) => [
                styles.centerButton,
                pressed && styles.centerPressed,
              ]}
            >
              <View style={styles.centerGlow} />
              <LinearGradient
                colors={['#A78BFA', '#7C3AED']}
                style={styles.centerGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              >
                <Ionicons name="sparkles" size={24} color="#FFFFFF" />
              </LinearGradient>
            </Pressable>
          );
        }

        return (
          <Pressable
            key={item.id}
            onPress={() => onTabPress?.(item.id)}
            style={({ pressed }) => [styles.tab, pressed && styles.tabPressed]}
          >
            {isActive ? (
              <View style={styles.activeCircle}>
                <Ionicons name={item.activeIcon} size={20} color="#FFFFFF" />
              </View>
            ) : (
              <Ionicons name={item.icon} size={23} color="#6B7280" />
            )}
          </Pressable>
        );
      })}
    </>
  );

  return (
    <View style={[styles.wrapper, { paddingBottom: Math.max(insets.bottom, 10) }]}>
      {Platform.OS === 'ios' ? (
        <BlurView intensity={40} tint="light" style={styles.container}>
          {content}
        </BlurView>
      ) : (
        <View style={[styles.container, styles.androidBar]}>{content}</View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: {
    paddingHorizontal: 16,
    backgroundColor: 'transparent',
  },
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    borderRadius: 32,
    paddingVertical: 10,
    paddingHorizontal: 8,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(237, 233, 254, 0.9)',
  },
  androidBar: {
    backgroundColor: 'rgba(255, 255, 255, 0.96)',
    shadowColor: '#8B5CF6',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 18,
    elevation: 12,
  },
  tab: {
    width: 48,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabPressed: {
    opacity: 0.7,
  },
  activeCircle: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: '#7C3AED',
    alignItems: 'center',
    justifyContent: 'center',
  },
  centerButton: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: -28,
  },
  centerGradient: {
    width: 58,
    height: 58,
    borderRadius: 29,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#7C3AED',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  centerPressed: {
    opacity: 0.88,
    transform: [{ scale: 0.95 }],
  },
  centerGlow: {
    position: 'absolute',
    width: 74,
    height: 74,
    borderRadius: 37,
    backgroundColor: 'rgba(139, 92, 246, 0.2)',
    zIndex: -1,
  },
});
