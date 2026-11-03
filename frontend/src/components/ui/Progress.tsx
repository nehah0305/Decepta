// What: This file implements the Progress module in frontend/src/components/ui/Progress.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface ProgressProps {
  value: number
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const Progress = ({ value }: ProgressProps) => (
  <div className="h-2 w-full overflow-hidden rounded-full bg-brand-bg/80">
    <div
      className="h-full rounded-full bg-gradient-to-r from-brand-primary to-brand-bright transition-all duration-500"
      style={{ width: `${value}%` }}
    ></div>
  </div>
)
