import type { ComponentType } from 'react';
import {
  View as RNView,
  Text as RNText,
  Pressable as RNPressable,
  type ViewProps,
  type TextProps,
  type PressableProps,
} from 'react-native';

/** Cast RN host components so React 19 JSX accepts style/props. */
export const View = RNView as unknown as ComponentType<ViewProps>;
export const Text = RNText as unknown as ComponentType<TextProps>;
export const Pressable = RNPressable as unknown as ComponentType<PressableProps>;
