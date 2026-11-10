import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  StyleSheet,
  StyleProp,
  ViewStyle,
  useWindowDimensions,
  Text,
} from 'react-native';
import LottieView from 'lottie-react-native';
import { Colors } from '../constants/colors';

const aiAssistantAnimation = require('../../assets/lottie/ai-assistant.json');

interface AIAnimationProps {
  size?: number;
  autoPlay?: boolean;
  loop?: boolean;
  style?: StyleProp<ViewStyle>;
}

export const AIAnimation: React.FC<AIAnimationProps> = ({
  size,
  autoPlay = true,
  loop = true,
  style,
}) => {
  const animationRef = useRef<LottieView>(null);
  const { width } = useWindowDimensions();
  const resolvedSize = size ?? Math.min(Math.max(width * 0.62, 210), 300);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (autoPlay && !failed) {
      animationRef.current?.play();
    } else {
      animationRef.current?.pause();
    }
  }, [autoPlay, failed]);

  if (failed) {
    return (
      <View style={[styles.container, { width: resolvedSize, height: resolvedSize }, style]}>
        <View style={[styles.fallback, { width: resolvedSize * 0.7, height: resolvedSize * 0.7, borderRadius: resolvedSize * 0.35 }]} />
        <Text style={styles.fallbackText}>AI</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { width: resolvedSize, height: resolvedSize }, style]}>
      <LottieView
        ref={animationRef}
        source={aiAssistantAnimation}
        autoPlay={autoPlay}
        loop={loop}
        style={{ width: resolvedSize, height: resolvedSize }}
        resizeMode="contain"
        enableSafeModeAndroid
        onAnimationFailure={() => setFailed(true)}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'visible',
  },
  fallback: {
    backgroundColor: Colors.primary[200],
    opacity: 0.85,
  },
  fallbackText: {
    position: 'absolute',
    color: Colors.primary[700],
    fontWeight: '700',
    fontSize: 18,
  },
});
