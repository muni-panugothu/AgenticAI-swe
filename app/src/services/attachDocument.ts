import { Alert, ActionSheetIOS, Platform } from 'react-native';
import { pickDocument } from './documentPicker';
import { ingestDocument } from './chatApi';

export type AttachChoice = 'file' | 'paste' | 'cancel';

export function showAttachOptions(
  onChoice: (choice: AttachChoice) => void
): void {
  if (Platform.OS === 'ios') {
    ActionSheetIOS.showActionSheetWithOptions(
      {
        options: ['Pick file (PDF/txt)', 'Paste text', 'Cancel'],
        cancelButtonIndex: 2,
      },
      (index) => {
        if (index === 0) onChoice('file');
        else if (index === 1) onChoice('paste');
        else onChoice('cancel');
      }
    );
    return;
  }

  Alert.alert('Add document', 'How would you like to add content?', [
    { text: 'Pick file (PDF/txt)', onPress: () => onChoice('file') },
    { text: 'Paste text', onPress: () => onChoice('paste') },
    { text: 'Cancel', style: 'cancel', onPress: () => onChoice('cancel') },
  ]);
}

export async function pickAndUploadFile(): Promise<{
  filename: string;
  chunkCount: number;
  strategy: string;
}> {
  const asset = await pickDocument();
  if (!asset) {
    throw new Error('No file selected');
  }

  if (__DEV__) {
    console.log('[upload] picker result:', {
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType,
    });
  }

  const { uploadDocument } = await import('./chatApi');
  const result = await uploadDocument(asset.uri, asset.name, asset.mimeType);
  return {
    filename: result.filename,
    chunkCount: result.chunkCount,
    strategy: result.strategy,
  };
}

export async function pasteAndIngest(text: string): Promise<{
  chunkCount: number;
  strategy: string;
}> {
  const result = await ingestDocument(text, 'recursive');
  return {
    chunkCount: result.chunkCount,
    strategy: 'recursive',
  };
}
