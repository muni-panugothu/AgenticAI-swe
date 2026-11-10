import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import { Colors } from '../constants/colors';

interface AIOrbProps {
  size?: number;
  variant?: 'home' | 'voice';
  animate?: boolean;
}

export const AIOrb: React.FC<AIOrbProps> = ({
  size = 200,
  variant = 'home',
  animate = true,
}) => {
  const scale = useSharedValue(1);
  const opacity = useSharedValue(0.6);

  useEffect(() => {
    if (animate) {
      scale.value = withRepeat(
        withTiming(1.05, { duration: 3000, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );
      opacity.value = withRepeat(
        withTiming(0.8, { duration: 3000, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );
    }
  }, [animate]);

  const animatedGlow = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  const isVoice = variant === 'voice';

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      {/* Outer glow */}
      <Animated.View
        style={[
          styles.glowOuter,
          {
            width: size * 1.4,
            height: size * 1.4,
            borderRadius: size * 0.7,
          },
          animatedGlow,
        ]}
      >
        <LinearGradient
          colors={
            isVoice
              ? ['rgba(196, 181, 253, 0.3)', 'rgba(221, 214, 254, 0.15)', 'transparent']
              : ['rgba(139, 92, 246, 0.25)', 'rgba(196, 181, 253, 0.15)', 'transparent']
          }
          style={StyleSheet.absoluteFill}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 1 }}
        />
      </Animated.View>

      {/* Mid glow */}
      <View
        style={[
          styles.glowMid,
          {
            width: size * 1.15,
            height: size * 1.15,
            borderRadius: size * 0.575,
          },
        ]}
      >
        <LinearGradient
          colors={
            isVoice
              ? ['rgba(237, 229, 255, 0.5)', 'rgba(248, 245, 255, 0.3)', 'transparent']
              : ['rgba(167, 139, 250, 0.2)', 'rgba(196, 181, 253, 0.1)', 'transparent']
          }
          style={StyleSheet.absoluteFill}
          start={{ x: 0.3, y: 0 }}
          end={{ x: 0.7, y: 1 }}
        />
      </View>

      {/* Main orb */}
      <View style={[styles.orbMain, { width: size, height: size, borderRadius: size / 2 }]}>
        <LinearGradient
          colors={
            isVoice
              ? ['#E8E0F0', '#D4C8E8', '#C8BCD8', '#DDD6E8']
              : ['#A78BFA', '#8B5CF6', '#7C3AED', '#9B7DFF']
          }
          style={[StyleSheet.absoluteFill, { borderRadius: size / 2 }]}
          start={{ x: 0.2, y: 0.1 }}
          end={{ x: 0.8, y: 0.9 }}
        />

        {/* Inner highlight - top */}
        <View
          style={[
            styles.highlight,
            {
              width: size * 0.7,
              height: size * 0.5,
              top: size * 0.08,
              left: size * 0.1,
              borderRadius: size * 0.35,
            },
          ]}
        >
          <LinearGradient
            colors={
              isVoice
                ? ['rgba(255, 255, 255, 0.6)', 'rgba(255, 255, 255, 0.1)', 'transparent']
                : ['rgba(255, 255, 255, 0.35)', 'rgba(255, 255, 255, 0.05)', 'transparent']
            }
            style={[StyleSheet.absoluteFill, { borderRadius: size * 0.35 }]}
            start={{ x: 0.3, y: 0 }}
            end={{ x: 0.5, y: 1 }}
          />
        </View>

        {/* Color accent - pink/blue */}
        {!isVoice && (
          <>
            <View
              style={[
                styles.colorAccent,
                {
                  width: size * 0.45,
                  height: size * 0.45,
                  bottom: size * 0.1,
                  right: size * 0.05,
                  borderRadius: size * 0.225,
                },
              ]}
            >
              <LinearGradient
                colors={['rgba(217, 70, 239, 0.3)', 'rgba(96, 165, 250, 0.3)', 'transparent']}
                style={[StyleSheet.absoluteFill, { borderRadius: size * 0.225 }]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
              />
            </View>
            <View
              style={[
                styles.colorAccent,
                {
                  width: size * 0.35,
                  height: size * 0.35,
                  bottom: size * 0.15,
                  left: size * 0.1,
                  borderRadius: size * 0.175,
                },
              ]}
            >
              <LinearGradient
                colors={['rgba(96, 165, 250, 0.3)', 'rgba(139, 92, 246, 0.15)', 'transparent']}
                style={[StyleSheet.absoluteFill, { borderRadius: size * 0.175 }]}
                start={{ x: 0, y: 0.5 }}
                end={{ x: 1, y: 0.5 }}
              />
            </View>
          </>
        )}

        {/* Voice variant inner rings */}
        {isVoice && (
          <>
            <View
              style={[
                styles.voiceRing,
                {
                  width: size * 0.6,
                  height: size * 0.6,
                  borderRadius: size * 0.3,
                  top: size * 0.2,
                  left: size * 0.2,
                },
              ]}
            >
              <LinearGradient
                colors={['rgba(200, 180, 220, 0.4)', 'rgba(220, 200, 240, 0.2)', 'transparent']}
                style={[StyleSheet.absoluteFill, { borderRadius: size * 0.3 }]}
                start={{ x: 0.3, y: 0.2 }}
                end={{ x: 0.7, y: 0.8 }}
              />
            </View>
            <View
              style={[
                styles.voiceRing,
                {
                  width: size * 0.35,
                  height: size * 0.35,
                  borderRadius: size * 0.175,
                  top: size * 0.32,
                  left: size * 0.32,
                },
              ]}
            >
              <LinearGradient
                colors={['rgba(180, 160, 200, 0.35)', 'rgba(200, 180, 220, 0.15)']}
                style={[StyleSheet.absoluteFill, { borderRadius: size * 0.175 }]}
                start={{ x: 0.4, y: 0.2 }}
                end={{ x: 0.6, y: 0.8 }}
              />
            </View>
          </>
        )}

        {/* Glass rim */}
        <View
          style={[
            styles.rim,
            {
              width: size,
              height: size,
              borderRadius: size / 2,
              borderWidth: 1,
              borderColor: isVoice
                ? 'rgba(255, 255, 255, 0.5)'
                : 'rgba(255, 255, 255, 0.2)',
            },
          ]}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  glowOuter: {
    position: 'absolute',
    overflow: 'hidden',
  },
  glowMid: {
    position: 'absolute',
    overflow: 'hidden',
  },
  orbMain: {
    overflow: 'hidden',
  },
  highlight: {
    position: 'absolute',
    overflow: 'hidden',
  },
  colorAccent: {
    position: 'absolute',
    overflow: 'hidden',
  },
  voiceRing: {
    position: 'absolute',
    overflow: 'hidden',
  },
  rim: {
    position: 'absolute',
  },
});
