import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Circle, Path, Rect } from 'react-native-svg';
import { AuthTheme } from '../constants/authTheme';

const ACCENT = AuthTheme.accent;

export const PersonFieldIcon: React.FC = () => (
  <View style={styles.wrap}>
    <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
      <Circle cx={12} cy={8} r={3.4} stroke={ACCENT} strokeWidth={1.8} />
      <Path
        d="M5.5 19.2c1.6-3.2 4-4.8 6.5-4.8s4.9 1.6 6.5 4.8"
        stroke={ACCENT}
        strokeWidth={1.8}
        strokeLinecap="round"
      />
    </Svg>
  </View>
);

export const EmailFieldIcon: React.FC = () => (
  <View style={styles.wrap}>
    <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
      <Rect
        x={3}
        y={5.5}
        width={18}
        height={13}
        rx={2.2}
        stroke={ACCENT}
        strokeWidth={1.8}
      />
      <Path
        d="M4.2 7.6L12 13.2L19.8 7.6"
        stroke={ACCENT}
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  </View>
);

/** Three lime dots with underline — matches reference password glyph. */
export const PasswordFieldIcon: React.FC = () => (
  <View style={styles.passwordWrap}>
    <View style={styles.dotsRow}>
      <View style={styles.dot} />
      <View style={styles.dot} />
      <View style={styles.dot} />
    </View>
    <View style={styles.underline} />
  </View>
);

const styles = StyleSheet.create({
  wrap: {
    width: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  passwordWrap: {
    width: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
  },
  dotsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
  },
  dot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: ACCENT,
  },
  underline: {
    marginTop: 3,
    width: 20,
    height: 1.5,
    borderRadius: 1,
    backgroundColor: ACCENT,
  },
});
