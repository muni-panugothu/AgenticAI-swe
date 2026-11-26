import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Path } from 'react-native-svg';
import { AssistantTheme } from '../../constants/assistantTheme';

/** Abstract interlocking mark inspired by the reference home logo. */
export const BrandMark: React.FC<{ size?: number }> = ({ size = 72 }) => (
  <View style={[styles.wrap, { width: size, height: size }]}>
    <Svg width={size} height={size} viewBox="0 0 72 72" fill="none">
      <Path
        d="M18 20c0-6 5-11 12-11h8c10 0 16 7 16 16v6c0 5-2 9-7 11l-9 4c-3 1-5 4-5 7v5c0 2 1.5 3.5 3.5 3.5H48"
        stroke={AssistantTheme.black}
        strokeWidth={7}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path
        d="M54 52c0 6-5 11-12 11h-8c-10 0-16-7-16-16v-6c0-5 2-9 7-11l9-4c3-1 5-4 5-7v-5c0-2-1.5-3.5-3.5-3.5H24"
        stroke={AssistantTheme.black}
        strokeWidth={7}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  </View>
);

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
  },
});
