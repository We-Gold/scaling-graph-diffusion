import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Reduce file watching by excluding unnecessary directories
  webpack: (config, { isServer }) => {
    // Ignore watching large directories
    config.watchOptions = {
      ...config.watchOptions,
      ignored: [
        '**/node_modules/**',
        '**/.git/**',
        '**/digress_env/**',
        '**/npm-local/**',
        '**/outputs/**',
        '**/DiGress/**',
        '**/archive/**',
        '**/scratch/**',
        '**/.next/**',
      ],
    };
    return config;
  },
};

export default nextConfig;
