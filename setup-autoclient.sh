#!/bin/bash

# First, ensure correct Node.js version is installed
echo "Installing correct Node.js version..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install 18.17.0
nvm use 18.17.0

# Create Next.js project with App Router
echo "Creating Next.js project..."
npx create-next-app@latest autoclient-frontend \
  --typescript \
  --tailwind \
  --app \
  --src-dir \
  --import-alias "@/*" \
  --use-yarn

# Navigate into project directory
cd autoclient-frontend

# Install dependencies using yarn
echo "Installing dependencies..."
yarn add clsx tailwind-merge @radix-ui/react-slot class-variance-authority lucide-react
yarn add @tanstack/react-table @radix-ui/react-select @radix-ui/react-slider
yarn add @radix-ui/react-switch @radix-ui/react-progress @radix-ui/react-dialog
yarn add @radix-ui/react-toast @radix-ui/react-accordion @radix-ui/react-hover-card
yarn add @radix-ui/react-tabs @radix-ui/react-scroll-area @radix-ui/react-separator
yarn add @radix-ui/react-label @radix-ui/react-checkbox @radix-ui/react-radio-group

# Initialize Shadcn UI
echo "Initializing Shadcn UI..."
npx shadcn-ui@latest init -y

# Add Shadcn components
echo "Adding Shadcn components..."
for component in button card input slider switch select progress table form tabs dialog toast accordion hover-card command sheet scroll-area separator label checkbox radio-group; do
  npx shadcn-ui@latest add $component -y
done

# Create required directories
echo "Creating directory structure..."
mkdir -p src/app/{manual-search,bulk-send,view-leads,search-terms,email-templates,projects,knowledge-base,autoclient-ai,automation,email-logs,settings,sent-campaigns}

# Install additional dependencies
echo "Installing additional dependencies..."
yarn add @tanstack/react-query axios zod react-hook-form @hookform/resolvers
yarn add -D @types/node @types/react @types/react-dom eslint-config-next

# Update next.config.js
echo "Configuring Next.js..."
cat > next.config.js << 'EOL'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: true,
  },
}

module.exports = nextConfig
EOL

# Create .env file
echo "Creating environment file..."
cat > .env.local << 'EOL'
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
EOL

echo "Setup complete! To start development:"
echo "cd autoclient-frontend"
echo "yarn dev" 