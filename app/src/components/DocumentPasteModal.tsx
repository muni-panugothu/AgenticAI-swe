import React, { useState } from 'react';
import {
  Modal,
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';

interface DocumentPasteModalProps {
  visible: boolean;
  onClose: () => void;
  onSubmit: (text: string) => void;
  loading?: boolean;
}

export function DocumentPasteModal({
  visible,
  onClose,
  onSubmit,
  loading = false,
}: DocumentPasteModalProps) {
  const [text, setText] = useState('');

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    onSubmit(trimmed);
    setText('');
  };

  const handleClose = () => {
    setText('');
    onClose();
  };

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={handleClose}>
      <KeyboardAvoidingView
        style={styles.overlay}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <View style={styles.sheet}>
          <Text style={styles.title}>Paste document text</Text>
          <Text style={styles.subtitle}>
            Paste article, notes, or PDF text here. It will be chunked and indexed for chat.
          </Text>
          <TextInput
            style={styles.input}
            multiline
            placeholder="Paste your document content…"
            placeholderTextColor="#A1A1AA"
            value={text}
            onChangeText={setText}
            editable={!loading}
            textAlignVertical="top"
          />
          <View style={styles.actions}>
            <Pressable onPress={handleClose} style={styles.cancelBtn} disabled={loading}>
              <Text style={styles.cancelText}>Cancel</Text>
            </Pressable>
            <Pressable
              onPress={handleSubmit}
              style={[styles.submitBtn, loading && styles.disabled]}
              disabled={loading || !text.trim()}
            >
              <Text style={styles.submitText}>{loading ? 'Uploading…' : 'Index text'}</Text>
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#FFFFFF',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 32 : 24,
    maxHeight: '80%',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1A1A2E',
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 13,
    color: '#6B7280',
    marginBottom: 14,
    lineHeight: 18,
  },
  input: {
    minHeight: 160,
    maxHeight: 280,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 14,
    padding: 14,
    fontSize: 14,
    color: '#1F1F2E',
    backgroundColor: '#FAFAFF',
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 10,
    marginTop: 16,
  },
  cancelBtn: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  cancelText: {
    fontSize: 15,
    color: '#6B7280',
    fontWeight: '600',
  },
  submitBtn: {
    backgroundColor: '#7C3AED',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 12,
  },
  submitText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },
  disabled: {
    opacity: 0.6,
  },
});
