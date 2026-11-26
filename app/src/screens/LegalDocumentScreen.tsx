import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AuthTheme, LEGAL_CONTENT } from '../constants/authTheme';
import { AuthChrome } from '../components/AuthChrome';

interface LegalDocumentScreenProps {
  kind: 'terms' | 'privacy';
  onBack: () => void;
}

export const LegalDocumentScreen: React.FC<LegalDocumentScreenProps> = ({
  kind,
  onBack,
}) => {
  const title =
    kind === 'terms' ? LEGAL_CONTENT.termsTitle : LEGAL_CONTENT.privacyTitle;
  const body = kind === 'terms' ? LEGAL_CONTENT.terms : LEGAL_CONTENT.privacy;

  return (
    <AuthChrome>
      <View style={styles.header}>
        <Pressable style={styles.backBtn} onPress={onBack} hitSlop={12}>
          <Ionicons
            name="arrow-undo-outline"
            size={26}
            color={AuthTheme.textPrimary}
          />
        </Pressable>
        <Text style={styles.heading}>{title}</Text>
      </View>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.body}>{body}</Text>
      </ScrollView>
    </AuthChrome>
  );
};

const styles = StyleSheet.create({
  header: {
    paddingHorizontal: 28,
    paddingTop: 24,
  },
  backBtn: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    marginBottom: 8,
  },
  heading: {
    color: AuthTheme.textPrimary,
    fontSize: 28,
    fontWeight: '800',
    marginBottom: 12,
  },
  scroll: {
    paddingHorizontal: 28,
    paddingBottom: 40,
  },
  body: {
    color: AuthTheme.textSecondary,
    fontSize: 14,
    lineHeight: 22,
  },
});
