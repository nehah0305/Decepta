import { Outlet } from 'react-router-dom'
import { MobileNav } from './MobileNav'
import { Sidebar } from './Sidebar'

export const DashboardLayout = () => (
  <div className="min-h-screen bg-brand-bg">
    <div className="absolute inset-0 pointer-events-none bg-grid-overlay opacity-40"></div>
    <div className="relative flex min-h-screen">
      <Sidebar />
      <div className="flex-1">
        <MobileNav />
        <main className="p-4 sm:p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  </div>
)
