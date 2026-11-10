import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '../constants/colors';

export interface CitationItem {
  index: number;
  source?: string;
  documentId: string;
  section?: string;
  excerpt?: string;
}

interface SourceCitationsProps {
  citations: CitationItem[];
}

export function SourceCitations({ citations }: SourceCitationsProps) {
  if (!citations.length) return null;

  return (
    <View style={styles.wrap}>
      <Text style={styles.title}>Sources</Text>
      {citations.map((c) => (
        <Text key={`${c.documentId}-${c.index}`} style={styles.line}>
          [{c.index}] {c.source ?? c.documentId}
          {c.section ? ` — ${c.section}` : ''}
        </Text>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginTop: 10,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: Colors.border.subtle,
  },
  title: {
    fontSize: 11,
    fontWeight: '700',
    color: Colors.text.secondary,
    marginBottom: 4,
  },
  line: {
    fontSize: 10,
    color: Colors.text.muted,
    lineHeight: 15,
    marginBottom: 2,
  },
});
