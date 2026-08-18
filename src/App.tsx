import { ReactNode } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { ToastViewport } from './components/ui/Toast'
import { useAuth } from './hooks/useAuth'
import { AuthLayout } from './layouts/AuthLayout'
import { DashboardLayout } from './layouts/DashboardLayout'
import { Analysis } from './pages/Analysis'
import { Detect } from './pages/Detect'
import { History } from './pages/History'
import { Landing } from './pages/Landing'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { Settings } from './pages/Settings'
import { User } from './pages/User'

const ProtectedRoute = ({ children }: { children: ReactNode }) => {
  const { isAuthenticated } = useAuth()
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return children
}

const PublicOnlyRoute = ({ children }: { children: ReactNode }) => {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <Navigate to="/detect" replace /> : children
}

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Landing />} />

        <Route
          element={
            <PublicOnlyRoute>
              <AuthLayout />
            </PublicOnlyRoute>
          }
        >
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
        </Route>

        <Route
          element={
            <ProtectedRoute>
              <DashboardLayout />
            </ProtectedRoute>
          }
        >
          <Route path="/detect" element={<Detect />} />
          <Route path="/history" element={<History />} />
          <Route path="/analysis/:id" element={<Analysis />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/user" element={<User />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastViewport />
    </>
  )
}

export default App
