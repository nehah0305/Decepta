# Depecta - Deepfake Detection Platform

A modern web application built with **React 19**, **TypeScript**, and **Vite** for detecting and analyzing deepfake content. The platform provides an intuitive dashboard for users to upload media files, perform deepfake detection, view analysis results, and track detection history.

## 🎯 Features

- **User Authentication**: Secure login and registration system with protected routes
- **File Upload & Detection**: Upload media files for deepfake analysis
- **Real-time Analysis**: View detection results with confidence scores and metrics
- **Detection History**: Track all previous detections and analyses
- **User Dashboard**: Comprehensive dashboard with detection cards and metrics
- **Settings Management**: Customize application preferences and themes
- **User Profile**: Manage user information and profile settings
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Type-Safe**: Full TypeScript support for reliability

## 🚀 Quick Start

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

```bash
# Clone or extract the project
cd depecta

# Install dependencies
npm install
```

### Development

```bash
# Start the development server with hot module replacement
npm run dev
```

The application will be available at `http://localhost:5173`

### Building

```bash
# Build for production
npm run build

# Preview the production build
npm run preview
```

### Linting

```bash
# Run Oxlint for code quality checks
npm run lint
```

## 📁 Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── analysis/       # Detection analysis components
│   ├── layout/         # Layout components (Sidebar, Navigation)
│   └── ui/             # Base UI components (Button, Card, Input, etc.)
├── pages/              # Page components
│   ├── Landing.tsx     # Home/landing page
│   ├── Login.tsx       # Login page
│   ├── Register.tsx    # User registration
│   ├── Detect.tsx      # File upload & detection
│   ├── History.tsx     # Detection history
│   ├── Analysis.tsx    # Detailed analysis view
│   ├── Settings.tsx    # Application settings
│   └── User.tsx        # User profile management
├── layouts/            # Layout wrappers
│   ├── AuthLayout.tsx  # Layout for auth pages
│   └── DashboardLayout.tsx # Main dashboard layout
├── hooks/              # Custom React hooks
│   ├── useAuth.tsx     # Authentication context & logic
│   ├── useDetection.tsx # Detection logic & state
│   └── useToast.tsx    # Toast notifications
├── types/              # TypeScript type definitions
├── data/               # Mock data and constants
├── App.tsx             # Main app component with routing
├── main.tsx            # Application entry point
└── index.css           # Global styles (Tailwind CSS)
```

## 🏗️ Architecture

### State Management
- **Context API**: Used for global state management (Auth, Detection, Toast)
- **Local Storage**: Persists user settings and detection history

### Routing
- **React Router v7**: Client-side routing with protected routes
- Public routes for landing, login, and register
- Protected routes for dashboard and analysis pages

### Styling
- **Tailwind CSS v4**: Utility-first CSS framework
- **Lucide React**: Icon library for UI components

### Type Safety
- **TypeScript 6.0**: Strict type checking with `verbatimModuleSyntax` enabled
- Custom type definitions in `src/types/`

## 🔒 Authentication Flow

1. Users land on the homepage (Landing page)
2. Can register a new account or login with existing credentials
3. After authentication, redirected to the Detect page
4. Authentication state is managed via `useAuth` hook
5. Protected routes check authentication status and redirect to login if needed

## 📊 Detection Flow

1. User uploads a media file via the Detect page
2. File is processed by the detection engine
3. Analysis results are displayed with:
   - Confidence score
   - Detection metrics
   - Timeline of analysis
4. Results are saved to history
5. User can view detailed analysis by clicking on a history item

## 🛠️ Build Configuration

### Vite
- **Plugin**: `@vitejs/plugin-react` with Oxc for fast transpilation
- **Tailwind CSS Vite**: Direct integration for optimized styles

### TypeScript
- **Strict Mode**: Enabled for maximum type safety
- **Module Resolution**: `bundler` for modern module handling
- **ESNext Target**: Uses modern JavaScript features

### Oxlint
- Code quality linting with React-specific rules
- Configured to warn about component export best practices

## 📦 Dependencies

- **react** (v19.2.8): UI library
- **react-dom** (v19.2.8): DOM rendering
- **react-router-dom** (v7.18.2): Client-side routing
- **tailwindcss** (v4.3.3): CSS framework
- **@tailwindcss/vite** (v4.3.3): Vite integration
- **lucide-react** (v1.31.0): Icon library

## 🧪 Code Quality

The project includes:
- ✅ TypeScript strict mode enabled
- ✅ Build passes with no errors
- ✅ Oxlint warnings for best practices
- ✅ React Fast Refresh support for development

## 📝 License

This project is private and confidential.

## 🤝 Contributing

For contributions, please follow the established code structure and maintain TypeScript type safety throughout the application.
