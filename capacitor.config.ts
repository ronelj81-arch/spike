import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.spikemarkets.signals',
  appName: 'Spike Market Signals',
  webDir: 'mobile-web',
  server: {
    // Keep the mobile app connected to the live FastAPI dashboard and its API.
    url: 'https://spike-market-signals.onrender.com',
    cleartext: false,
    allowNavigation: ['spike-market-signals.onrender.com'],
  },
  android: {
    allowMixedContent: false,
  },
  ios: {
    contentInset: 'automatic',
  },
};

export default config;
