import { Outlet } from 'react-router-dom'
import { MobileNav } from './MobileNav'
import { Sidebar } from './Sidebar'

export const DashboardLayout = () => (
  <div className="min-h-screen" style={{
    background: 'linear-gradient(135deg, #0C111D 0%, #142633 50%, #0C111D 100%), radial-gradient(circle at 12% 16%, rgba(38, 184, 181, 0.16), transparent 40%), radial-gradient(circle at 88% 7%, rgba(80, 214, 209, 0.14), transparent 32%), radial-gradient(circle at 50% 100%, rgba(38, 184, 181, 0.08), transparent 50%)',
    backgroundAttachment: 'fixed'
  }}>
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-20"></div>
    
    {/* Animated floating accents */}
    <div className="pointer-events-none fixed top-1/4 right-1/4 h-96 w-96 rounded-full bg-brand-bright/3 blur-3xl animate-glow-drift" style={{animationDelay: '0s'}}></div>
    <div className="pointer-events-none fixed bottom-1/4 left-1/3 h-80 w-80 rounded-full bg-brand-primary/3 blur-3xl animate-pulse-glow" style={{animationDelay: '2s'}}></div>
    
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
