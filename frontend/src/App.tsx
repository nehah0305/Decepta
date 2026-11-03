// What: This file implements the App module in frontend/src/App.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import type { ReactNode } from 'react'
import { Navigate, Route, Routes, useLocation, Outlet } from 'react-router-dom'
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
import { Models } from './pages/Models'

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const ProtectedRoute = ({ children }: { children?: ReactNode }) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const { isAuthenticated } = useAuth()
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return children ? <>{children}</> : <Outlet />
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const PublicOnlyRoute = ({ children }: { children: ReactNode }) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <Navigate to="/detect" replace /> : children
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
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

        <Route element={<DashboardLayout />}>
          {/* Publicly accessible Model Architecture Explorer & Benchmark */}
          <Route path="/models" element={<Models />} />

          {/* Protected forensic workspace routes */}
          <Route
            element={
              <ProtectedRoute>
                <Outlet />
              </ProtectedRoute>
            }
          >
            <Route path="/detect" element={<Detect />} />
            <Route path="/history" element={<History />} />
            <Route path="/analysis/:id" element={<Analysis />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/user" element={<User />} />
          </Route>
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastViewport />
    </>
  )
}

export default App
