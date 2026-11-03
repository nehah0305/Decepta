// What: This file implements the MobileNav module in frontend/src/components/layout/MobileNav.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { primaryNav, secondaryNav } from './Sidebar'

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const MobileNav = () => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [open, setOpen] = useState(false)

  return (
    <header className="sticky top-0 z-30 border-b border-brand-border bg-brand-surface/90 px-4 py-3 backdrop-blur lg:hidden">
      <div className="flex items-center justify-between">
        <p className="text-base font-orbitron font-bold text-brand-text gradient-text">DECEPTA</p>
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
