export interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: string;
  color: string;
}

export const quickActions: QuickAction[] = [
  {
    id: 'write',
    title: 'Write Content',
    description: 'Blog, emails, social posts & more',
    icon: 'pencil-outline',
    color: '#7C3AED',
  },
  {
    id: 'summarize',
    title: 'Summarize',
    description: 'Articles, pdf & docs',
    icon: 'clipboard-text-outline',
    color: '#7C3AED',
  },
  {
    id: 'brainstorm',
    title: 'Brainstorm',
    description: 'Ideas, strategies and more',
    icon: 'bullhorn-outline',
    color: '#7C3AED',
  },
  {
    id: 'code',
    title: 'Code Assistant',
    description: 'Write, debug & explain code',
    icon: 'code-tags',
    color: '#7C3AED',
  },
];
