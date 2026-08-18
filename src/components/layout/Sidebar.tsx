import { BarChart3, History, House, Settings, User } from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface NavItem {
  label: string
  to: string
  icon: typeof House
}

export const primaryNav: NavItem[] = [
  { label: 'Home', to: '/detect', icon: House },
  { label: 'History', to: '/history', icon: History },
  { label: 'Analysis', to: '/analysis/latest', icon: BarChart3 },
]

export const secondaryNav: NavItem[] = [
  { label: 'Settings', to: '/settings', icon: Settings },
  { label: 'User', to: '/user', icon: User },
]

const itemClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
    isActive
      ? 'bg-brand-primary/18 text-brand-bright shadow-[inset_0_0_0_1px_rgba(80,214,209,0.28)]'
      : 'text-brand-subtle hover:bg-brand-card2/60 hover:text-brand-text'
  }`

export const Sidebar = () => (
  <aside className="hidden h-screen w-[232px] flex-col border-r border-brand-border bg-brand-surface/70 p-5 backdrop-blur-md lg:flex">
    <p className="text-lg font-semibold tracking-wide text-brand-text">Project Name</p>

    <nav className="mt-7 space-y-1">
      {primaryNav.map(({ label, to, icon: Icon }) => (
        <NavLink key={label} to={to} className={itemClass}>
          <Icon className="h-4 w-4" />
          {label}
        </NavLink>
      ))}
    </nav>

    <div className="my-5 h-px bg-brand-border"></div>

    <nav className="space-y-1">
      {secondaryNav.map(({ label, to, icon: Icon }) => (
        <NavLink key={label} to={to} className={itemClass}>
          <Icon className="h-4 w-4" />
          {label}
        </NavLink>
      ))}
    </nav>
  </aside>
)
