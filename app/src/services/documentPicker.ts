/**
 * Lazy-load expo-document-picker so the app does not crash when the
 * native module is missing from an older dev-client build.
 * After adding expo-document-picker, rebuild: npm run android
 */
export interface PickedDocument {
  uri: string;
  name: string;
  mimeType?: string | null;
}

export async function pickDocument(): Promise<PickedDocument | null> {
  try {
    const DocumentPicker = await import('expo-document-picker');
    const result = await DocumentPicker.getDocumentAsync({
      type: [
        'application/pdf',
        'text/plain',
        'text/csv',
        'text/markdown',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/*',
      ],
      copyToCacheDirectory: true,
      multiple: false,
    });

    if (result.canceled || !result.assets?.[0]) return null;

    const asset = result.assets[0];
    return {
      uri: asset.uri,
      name: asset.name ?? asset.uri.split('/').pop() ?? 'document',
      mimeType: asset.mimeType,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (
      message.includes('ExpoDocumentPicker') ||
      message.includes('native module')
    ) {
      throw new Error(
        'Document picker requires a dev-client rebuild.\n\n' +
          'Run from repo root:\n' +
          '  npm run android\n\n' +
          'Then reopen the app on your device.'
      );
    }
    throw err;
  }
}
