import { Outlet } from 'react-router-dom'

export const AuthLayout = () => (
  <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-brand-bg px-4">
    <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_10%_20%,rgba(38,184,181,0.18),transparent_36%),radial-gradient(circle_at_90%_12%,rgba(80,214,209,0.2),transparent_40%),linear-gradient(135deg,rgba(17,58,64,0.3),transparent)]"></div>
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-45"></div>
    <div className="relative w-full max-w-md">
      <Outlet />
    </div>
  </div>
)
