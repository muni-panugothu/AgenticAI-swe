import React from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import { Colors } from '../constants/colors';
import { MessageActions } from './MessageActions';

interface AIMessageCardProps {
  text: string;
  imageUri?: string;
  showActions?: boolean;
}

export const AIMessageCard: React.FC<AIMessageCardProps> = ({
  text,
  imageUri,
  showActions = true,
}) => {
  return (
    <View style={styles.container}>
      <View style={styles.bubble}>
        <Text style={styles.text}>{text}</Text>
        {imageUri && (
          <View style={styles.imageContainer}>
            <Image
              source={{ uri: imageUri }}
              style={styles.image}
              resizeMode="cover"
            />
          </View>
        )}
        {showActions && <MessageActions />}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'flex-start',
    marginBottom: 12,
    paddingHorizontal: 16,
  },
  bubble: {
    maxWidth: '85%',
    backgroundColor: Colors.surface,
    borderRadius: 18,
    borderBottomLeftRadius: 4,
    paddingHorizontal: 14,
    paddingTop: 12,
    paddingBottom: 8,
    borderWidth: 1,
    borderColor: Colors.border.light,
    shadowColor: Colors.shadow.card,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 1,
  },
  text: {
    fontSize: 13.5,
    fontWeight: '400',
    color: Colors.text.primary,
    lineHeight: 20,
    marginBottom: 10,
  },
  imageContainer: {
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 8,
  },
  image: {
    width: '100%',
    height: 160,
    borderRadius: 12,
    backgroundColor: Colors.primary[100],
  },
});
