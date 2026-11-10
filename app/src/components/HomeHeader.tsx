import React from 'react';
import { View, StyleSheet } from 'react-native';
import { ProfileAvatar } from './ProfileAvatar';
import { PremiumButton } from './PremiumButton';

interface HomeHeaderProps {
  onProfilePress?: () => void;
}

export const HomeHeader: React.FC<HomeHeaderProps> = ({ onProfilePress }) => {
  return (
    <View style={styles.container}>
      <ProfileAvatar size={44} onPress={onProfilePress} />
      <PremiumButton />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 4,
    paddingBottom: 8,
  },
});
