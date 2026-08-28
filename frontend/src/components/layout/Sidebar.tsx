import { BarChart3, History, House, Settings, User, ShieldCheck, Cpu, Sparkles } from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface NavItem {
  label: string
  to: string
  icon: typeof House
}

export const primaryNav: NavItem[] = [
  { label: 'Analyzer', to: '/detect', icon: House },
  { label: 'History & Logs', to: '/history', icon: History },
  { label: 'Detailed Report', to: '/analysis/latest', icon: BarChart3 },
]

export const secondaryNav: NavItem[] = [
  { label: 'System Settings', to: '/settings', icon: Settings },
  { label: 'User Profile', to: '/user', icon: User },
]

const itemClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-3.5 rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-200 ${
    isActive
      ? 'bg-brand-primary/20 text-brand-bright shadow-[inset_0_0_0_1px_rgba(80,214,209,0.38)] glow-effect-sm font-semibold'
      : 'text-brand-subtle hover:bg-brand-card2/60 hover:text-brand-text hover:translate-x-1'
  }`

export const Sidebar = () => (
  <aside className="hidden h-screen w-[256px] flex-col border-r border-brand-border bg-brand-surface/85 p-5 backdrop-blur-xl lg:flex justify-between shrink-0">
    <div>
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-2 py-2">
        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-bright via-brand-primary to-[#142633] text-brand-bg shadow-lg glow-effect">
          <ShieldCheck className="h-6 w-6 text-brand-bg" />
        </div>
        <div>
          <h1 className="font-orbitron text-2xl font-bold tracking-wider gradient-text">DECEPTA</h1>
          <p className="text-[10px] uppercase font-bold tracking-widest text-brand-bright/80 flex items-center gap-1">
            <Sparkles className="h-3 w-3" /> Visual AI Suite
          </p>
        </div>
      </div>

      <div className="mt-8 mb-3 px-2">
        <p className="text-[11px] font-bold uppercase tracking-widest text-brand-muted">Main Navigation</p>
      </div>

      <nav className="space-y-1.5">
        {primaryNav.map(({ label, to, icon: Icon }) => (
          <NavLink key={label} to={to} className={itemClass}>
            <Icon className="h-4.5 w-4.5 shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="my-6 h-px bg-gradient-to-r from-transparent via-brand-border to-transparent"></div>

      <div className="mb-3 px-2">
        <p className="text-[11px] font-bold uppercase tracking-widest text-brand-muted">Preferences</p>
      </div>

      <nav className="space-y-1.5">
        {secondaryNav.map(({ label, to, icon: Icon }) => (
          <NavLink key={label} to={to} className={itemClass}>
            <Icon className="h-4.5 w-4.5 shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>

    {/* Live Engine Status Card */}
    <div className="rounded-2xl border border-brand-border bg-brand-card/70 p-4 space-y-2.5 backdrop-blur-md glow-effect-sm">
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
  </aside>
)
