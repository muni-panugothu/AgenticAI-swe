import React, { useEffect, useRef } from 'react';
import { View, Animated, type ViewStyle } from 'react-native';

const BAR_COUNT = 28;
const HEIGHTS = [8, 14, 22, 12, 28, 16, 24, 10, 20, 30, 14, 26, 18, 12, 22];

interface VoiceWaveformProps {
  active: boolean;
}

export const VoiceWaveform: React.FC<VoiceWaveformProps> = ({ active }) => {
  const anims = useRef(
    Array.from({ length: BAR_COUNT }, () => new Animated.Value(0.35))
  ).current;

  useEffect(() => {
    if (!active) {
      anims.forEach((a) => a.stopAnimation());
      return;
    }

    const loops = anims.map((anim, index) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: 0.25 + (HEIGHTS[index % HEIGHTS.length] % 10) / 12,
            duration: 220 + (index % 5) * 40,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0.9,
            duration: 240 + (index % 4) * 35,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0.35,
            duration: 200 + (index % 6) * 30,
            useNativeDriver: true,
          }),
        ])
      )
    );

    loops.forEach((loop) => loop.start());
    return () => loops.forEach((loop) => loop.stop());
  }, [active, anims]);

  const wrapStyle: ViewStyle = {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    height: 40,
    marginHorizontal: 8,
    overflow: 'hidden',
  };

  return (
    <View style={wrapStyle}>
      {anims.map((anim, index) => (
        <Animated.View
          key={`bar-${index}`}
          style={{
            width: 3,
            height: HEIGHTS[index % HEIGHTS.length],
            borderRadius: 2,
            backgroundColor: '#8E8E8E',
            marginHorizontal: 1.5,
            transform: [{ scaleY: anim }],
            opacity: active ? 1 : 0.55,
          }}
        />
      ))}
    </View>
  );
};
