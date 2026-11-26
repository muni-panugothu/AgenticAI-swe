/** Premium light assistant UI (home + chat reference). */
export const AssistantTheme = {
  background: '#F4F4F2',
  surface: '#FFFFFF',
  textPrimary: '#111111',
  textSecondary: '#6B6B6B',
  textMuted: '#9A9A9A',
  border: '#E8E8E6',
  userBubble: '#ECECEB',
  headerPill: '#EFEFEE',
  overlay: 'rgba(0,0,0,0.28)',
  black: '#000000',
  icon: '#5C5C5C',
} as const;

export function greetingForNow(name: string): string {
  const hour = new Date().getHours();
  const period =
    hour < 12 ? 'Good Morning' : hour < 17 ? 'Good Afternoon' : 'Good Evening';
  return `${period}, ${name}`;
}

/** Display label for model chip — never invent a fake product model. */
export function resolveModelLabel(liveModel?: string | null): string {
  const fromEnv = process.env.EXPO_PUBLIC_CHAT_MODEL_LABEL?.trim();
  if (fromEnv) return fromEnv;
  if (liveModel && liveModel !== 'unknown') return liveModel;
  return 'AutoPilot';
}
