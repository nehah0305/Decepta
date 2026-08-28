import { Outlet, Link } from 'react-router-dom'
import { MobileNav } from './MobileNav'
import { Sidebar } from './Sidebar'
import { Bell, Search } from 'lucide-react'
import { useAuth } from '../../hooks/useAuth'

export const DashboardLayout = () => {
  const { user } = useAuth()

  return (
    <div className="min-h-screen relative bg-[#0C111D]">
      {/* Background Cyber Grid */}
      <div className="pointer-events-none fixed inset-0 bg-grid-overlay opacity-25"></div>
      
      {/* Ambient floating glow accents */}
      <div className="pointer-events-none fixed top-1/4 right-1/4 h-96 w-96 rounded-full bg-brand-bright/4 blur-3xl animate-glow-drift" style={{ animationDelay: '0s' }}></div>
      <div className="pointer-events-none fixed bottom-1/4 left-1/3 h-80 w-80 rounded-full bg-brand-primary/4 blur-3xl animate-float" style={{ animationDelay: '2s' }}></div>
      
      <div className="relative flex min-h-screen">
        <Sidebar />
        
        <div className="flex-1 flex flex-col min-w-0">
          <MobileNav />

          {/* Global Top Navbar */}
          <header className="hidden lg:flex items-center justify-between h-16 px-8 border-b border-brand-border/60 bg-brand-surface/70 backdrop-blur-xl sticky top-0 z-30">
            {/* Quick Search Shortcut */}
            <div className="flex items-center gap-3">
              <div className="relative w-72">
                <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-brand-muted" />
                <input
                  type="text"
                  placeholder="Quick search audits or models..."
                  className="w-full rounded-xl border border-brand-border/60 bg-brand-card/50 py-1.5 pl-9 pr-3 text-xs text-brand-text placeholder:text-brand-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
                />
              </div>
              <span className="text-[10px] font-mono font-semibold px-2 py-1 rounded bg-brand-card2/50 border border-brand-border text-brand-muted">
                ⌘K
              </span>
            </div>

            {/* Top Right Quick Actions */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span>GPU Acceleration Active</span>
              </div>

              {/* Notification Button */}
              <button
                title="Notifications"
                className="relative p-2 rounded-xl text-brand-subtle hover:text-brand-bright hover:bg-brand-card2/60 transition-colors"
              >
                <Bell className="h-4.5 w-4.5" />
                <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-brand-bright"></span>
              </button>

              <div className="h-5 w-px bg-brand-border"></div>

              {/* User Quick Link */}
              {user ? (
                <Link to="/user" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                  <div className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-brand-bright to-brand-primary text-xs font-bold text-brand-bg shadow-sm">
                    {user.name ? user.name.slice(0, 1).toUpperCase() : 'U'}
                  </div>
                  <span className="text-xs font-semibold text-brand-text truncate max-w-[120px]">
                    {user.name}
                  </span>
                </Link>
              ) : null}
            </div>
          </header>

          <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}
