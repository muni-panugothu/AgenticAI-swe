import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Colors } from '../constants/colors';
import { useAuth } from '../context/AuthContext';

export const ProfileScreen: React.FC = () => {
  const { user, signOut } = useAuth();
  const navigation = useNavigation();

  const displayName = user?.display_name || user?.email?.split('@')[0] || 'User';
  const initials = displayName.slice(0, 2).toUpperCase();

  return (
    <View style={styles.root}>
      <LinearGradient
        colors={['#FAFAFE', '#F7F7FD', '#F0EBFF']}
        style={StyleSheet.absoluteFill}
      />
      <SafeAreaView style={styles.safe} edges={['top']}>
        <Pressable onPress={() => navigation.goBack()} style={styles.back}>
          <Ionicons name="chevron-back" size={24} color={Colors.primary[600]} />
          <Text style={styles.backText}>Back</Text>
        </Pressable>
        <ScrollView contentContainerStyle={styles.content}>
          <LinearGradient
            colors={['#A78BFA', '#7C3AED']}
            style={styles.avatar}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <Text style={styles.avatarText}>{initials}</Text>
          </LinearGradient>
          <Text style={styles.name}>{displayName}</Text>
          <Text style={styles.email}>{user?.email}</Text>

          <View style={styles.infoCard}>
            <Row label="User ID" value={user?.id ?? '—'} mono />
            <Row label="Tenant" value={user?.tenant_id ?? '—'} mono />
            <Row label="Status" value={user?.status ?? 'active'} />
          </View>

          <Pressable
            style={({ pressed }) => [styles.logout, pressed && { opacity: 0.9 }]}
            onPress={async () => {
              await signOut();
              navigation.goBack();
            }}
          >
            <Text style={styles.logoutText}>Sign Out</Text>
          </Pressable>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
};

const Row: React.FC<{ label: string; value: string; mono?: boolean }> = ({
  label,
  value,
  mono,
}) => (
  <View style={styles.row}>
    <Text style={styles.rowLabel}>{label}</Text>
    <Text style={[styles.rowValue, mono && styles.mono]} numberOfLines={2}>
      {value}
    </Text>
  </View>
);

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },
  back: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  backText: {
    color: Colors.primary[600],
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 4,
  },
  content: {
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingBottom: 40,
  },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 12,
  },
  avatarText: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '700',
  },
  name: {
    marginTop: 16,
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text.primary,
  },
  email: {
    marginTop: 4,
    fontSize: 15,
    color: Colors.text.secondary,
  },
  infoCard: {
    width: '100%',
    marginTop: 28,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border.light,
    padding: 16,
    gap: 14,
  },
  row: { gap: 4 },
  rowLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.text.muted,
    textTransform: 'uppercase',
  },
  rowValue: {
    fontSize: 15,
    color: Colors.text.primary,
  },
  mono: { fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', fontSize: 13 },
  logout: {
    marginTop: 32,
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: Colors.border.light,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  logoutText: {
    color: '#DC2626',
    fontSize: 16,
    fontWeight: '600',
  },
});
