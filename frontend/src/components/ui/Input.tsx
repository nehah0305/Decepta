// What: This file implements the Input module in frontend/src/components/ui/Input.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import type { InputHTMLAttributes } from 'react'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string
  error?: string
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const Input = ({ label, id, error, className = '', ...props }: InputProps) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const inputId = id ?? label.toLowerCase().replace(/\s+/g, '-')

  return (
    <div className="space-y-1.5">
      <label htmlFor={inputId} className="text-sm text-brand-subtle">
        {label}
      </label>
      <input
        id={inputId}
        className={`w-full rounded-xl border border-brand-border bg-brand-card2/70 px-3.5 py-2.5 text-sm text-brand-text placeholder:text-brand-muted transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright ${error ? 'border-rose-400/50' : ''} ${className}`}
        {...props}
      />
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  )
}
