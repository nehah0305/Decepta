import { Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { primaryNav, secondaryNav } from './Sidebar'

export const MobileNav = () => {
  const [open, setOpen] = useState(false)

  return (
    <header className="sticky top-0 z-30 border-b border-brand-border bg-brand-surface/90 px-4 py-3 backdrop-blur lg:hidden">
      <div className="flex items-center justify-between">
        <p className="text-base font-semibold text-brand-text">Depecta</p>
        <button
          type="button"
          className="rounded-lg border border-brand-border bg-brand-card p-2 text-brand-text"
          aria-label="Toggle navigation menu"
          onClick={() => setOpen((prev) => !prev)}
        >
          {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      </div>

      {open ? (
        <nav className="mt-3 space-y-1">
          {[...primaryNav, ...secondaryNav].map(({ label, to, icon: Icon }) => (
            <NavLink
              key={label}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${
                  isActive ? 'bg-brand-primary/20 text-brand-bright' : 'text-brand-subtle'
                }`
              }
              onClick={() => setOpen(false)}
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      ) : null}
    </header>
  )
}
