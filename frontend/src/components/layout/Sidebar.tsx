import { BarChart3, History, House, Settings, User, ShieldCheck, Cpu, Sparkles, LogOut, Layers } from 'lucide-react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'

interface NavItem {
  label: string
  to: string
  icon: typeof House
}

export const primaryNav: NavItem[] = [
  { label: 'Deepfake Analyzer', to: '/detect', icon: House },
  { label: 'History & Audit Logs', to: '/history', icon: History },
  { label: 'Latest Report', to: '/analysis/latest', icon: BarChart3 },
  { label: 'Research & Benchmarks', to: '/models', icon: Layers },
]

export const secondaryNav: NavItem[] = [
  { label: 'System Settings', to: '/settings', icon: Settings },
  { label: 'User Profile', to: '/user', icon: User },
]

const itemClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-3.5 rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-200 ${
    isActive
      ? 'bg-brand-primary/22 text-brand-bright shadow-[inset_0_0_0_1px_rgba(80,214,209,0.4)] glow-effect-sm font-semibold'
      : 'text-brand-subtle hover:bg-brand-card2/60 hover:text-brand-text hover:translate-x-1'
  }`

export const Sidebar = () => {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  return (
    <aside className="sticky top-0 h-screen w-[265px] flex-col border-r border-brand-border bg-brand-surface/90 p-5 backdrop-blur-xl lg:flex justify-between shrink-0 overflow-y-auto z-40">
      <div className="space-y-6">
        {/* Brand Header */}
        <Link to="/detect" className="flex items-center gap-3 px-2 py-1.5 group cursor-pointer">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-bright via-brand-primary to-[#142633] text-brand-bg shadow-lg glow-effect group-hover:scale-105 transition-transform">
            <ShieldCheck className="h-6 w-6 text-brand-bg" />
          </div>
          <div>
            <h1 className="font-orbitron text-2xl font-bold tracking-wider gradient-text">DECEPTA</h1>
            <p className="text-[10px] uppercase font-bold tracking-widest text-brand-bright/90 flex items-center gap-1">
              <Sparkles className="h-3 w-3" /> Visual AI Suite
            </p>
          </div>
        </Link>

        {/* Primary Navigation */}
        <div className="space-y-1.5">
          <p className="px-2 text-[11px] font-bold uppercase tracking-widest text-brand-muted">Core Modules</p>
          <nav className="space-y-1.5">
            {primaryNav.map(({ label, to, icon: Icon }) => (
              <NavLink key={label} to={to} className={itemClass}>
                <Icon className="h-4.5 w-4.5 shrink-0" />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Secondary Navigation */}
        <div className="space-y-1.5">
          <p className="px-2 text-[11px] font-bold uppercase tracking-widest text-brand-muted">Configuration</p>
          <nav className="space-y-1.5">
            {secondaryNav.map(({ label, to, icon: Icon }) => (
              <NavLink key={label} to={to} className={itemClass}>
                <Icon className="h-4.5 w-4.5 shrink-0" />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      <div className="space-y-4 pt-6">
        {/* Live Engine Status Card */}
        <div className="rounded-2xl border border-brand-border/80 bg-brand-card/80 p-4 space-y-2.5 backdrop-blur-md glow-effect-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-bright opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-brand-bright"></span>
              </span>
              <span className="text-[11px] font-bold uppercase tracking-wider text-brand-bright">
                Engine Active
              </span>
            </div>
            <Cpu className="h-4 w-4 text-brand-bright" />
          </div>
          <p className="text-xs font-semibold text-brand-text truncate">ResNet-50 Fine-Tuned</p>
          <div className="flex items-center justify-between text-[11px] text-brand-subtle font-medium">
            <span>Validation ROC-AUC</span>
            <span className="font-orbitron font-bold text-brand-bright">72.88%</span>
          </div>
        </div>

        {/* User Account Quick Footer */}
        {user ? (
          <div className="flex items-center justify-between gap-3 rounded-2xl border border-brand-border/50 bg-brand-card2/40 p-3">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-brand-bright/30 to-brand-primary/30 text-sm font-bold text-brand-bright border border-brand-bright/40">
                {user.name ? user.name.slice(0, 1).toUpperCase() : 'U'}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-brand-text truncate">{user.name}</p>
                <p className="text-[10px] text-brand-muted truncate">{user.email}</p>
              </div>
            </div>
            <button
              onClick={() => {
                logout()
                navigate('/login')
              }}
              title="Logout"
              className="p-1.5 rounded-lg text-brand-subtle hover:text-red-400 hover:bg-red-500/15 transition-colors shrink-0"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        ) : null}
      </div>
    </aside>
  )
}
