import React from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';

interface ProfileAvatarProps {
  size?: number;
  onPress?: () => void;
}

export const ProfileAvatar: React.FC<ProfileAvatarProps> = ({
  size = 44,
  onPress,
}) => {
  const { user } = useAuth();
  const label =
    user?.display_name?.slice(0, 2) ||
    user?.email?.slice(0, 2).toUpperCase() ||
    null;

  const inner = label ? (
    <Text style={[styles.initials, { fontSize: size * 0.36 }]}>{label}</Text>
  ) : (
    <Ionicons name="person" size={size * 0.48} color="#FFFFFF" />
  );

  const avatar = (
    <LinearGradient
      colors={['#A78BFA', '#7C3AED']}
      style={[
        styles.container,
        { width: size, height: size, borderRadius: size / 2 },
      ]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      {inner}
    </LinearGradient>
  );

  if (!onPress) return avatar;
  return (
    <Pressable onPress={onPress} accessibilityRole="button">
      {avatar}
    </Pressable>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  initials: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
});
