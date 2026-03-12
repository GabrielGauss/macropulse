/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        surface: {
          0: '#0a0b0f',
          1: '#111318',
          2: '#1a1c23',
          3: '#23262f',
          4: '#2c3039',
        },
        accent: {
          green: '#00e5a0',
          red: '#ff4d6a',
          amber: '#ffb800',
          blue: '#3b82f6',
          purple: '#a78bfa',
        },
        regime: {
          expansion: '#00e5a0',
          tightening: '#ff4d6a',
          'risk-off': '#ffb800',
          recovery: '#3b82f6',
        },
      },
    },
  },
  plugins: [],
};
