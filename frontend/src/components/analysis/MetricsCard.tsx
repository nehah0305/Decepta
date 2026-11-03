// What: This file implements the MetricsCard module in frontend/src/components/analysis/MetricsCard.tsx.
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
interface MetricsCardProps {
  label: string
  value: string
  icon?: ReactNode
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const MetricsCard = ({ label, value, icon }: MetricsCardProps) => (
  <div className="rounded-2xl border border-brand-border bg-brand-card/60 p-4 hover:border-brand-bright/40 transition-colors space-y-1.5 backdrop-blur-sm">
    <div className="flex items-center justify-between">
      <p className="text-[11px] font-bold uppercase tracking-wider text-brand-muted">{label}</p>
      {icon ? <span className="text-brand-bright">{icon}</span> : null}
    </div>
    <p className="text-lg font-bold font-orbitron text-brand-text truncate">{value}</p>
  </div>
)
