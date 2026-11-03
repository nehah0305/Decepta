// What: This file implements the Card module in frontend/src/components/ui/Card.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import type { ReactNode } from 'react'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface CardProps {
  children: ReactNode
  className?: string
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const Card = ({ children, className = '' }: CardProps) => (
  <section
    className={`rounded-2xl border border-brand-border bg-brand-card/80 p-5 shadow-[0_20px_48px_rgba(5,18,22,0.55)] backdrop-blur-sm ${className}`}
  >
    {children}
  </section>
)
