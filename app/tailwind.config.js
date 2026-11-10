/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./App.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#F5F0FF',
          100: '#EDE5FF',
          200: '#DDD6FE',
          300: '#C4B5FD',
          400: '#A78BFA',
          500: '#8B5CF6',
          600: '#7C3AED',
          700: '#6D28D9',
          800: '#5B21B6',
          900: '#4C1D95',
        },
        surface: '#FFFFFF',
        background: '#F8F5FF',
        textPrimary: '#1A1A2E',
        textSecondary: '#6B7280',
        textMuted: '#9CA3AF',
        border: '#EDE5FF',
      },
    },
  },
  plugins: [],
};
