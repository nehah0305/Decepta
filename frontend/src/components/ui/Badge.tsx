// What: This file implements the Badge module in frontend/src/components/ui/Badge.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface BadgeProps {
  label: string
  tone?: 'success' | 'warning' | 'error' | 'neutral'
}

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const tones = {
  success: 'bg-emerald-500/20 text-emerald-200 border-emerald-300/35',
  warning: 'bg-amber-500/20 text-amber-200 border-amber-300/35',
  error: 'bg-red-500/20 text-red-200 border-red-300/35',
  neutral: 'bg-brand-card2/60 text-brand-subtle border-brand-border',
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const Badge = ({ label, tone = 'neutral' }: BadgeProps) => (
  <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${tones[tone]}`}>
    {label}
  </span>
)
