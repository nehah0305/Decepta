// What: This file implements the Toast module in frontend/src/components/ui/Toast.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react'
import { useToast } from '../../hooks/useToast'

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const iconMap = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const ToastViewport = () => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const { toasts, dismissToast } = useToast()

  return (
    <div className="pointer-events-none fixed right-4 top-4 z-50 flex w-full max-w-sm flex-col gap-2">
      {toasts.map((toast) => {
        /**
         * What: This declaration defines the named behavior or data shape.
         * How: It processes the values supplied by callers according to its implementation.
         * Input: Declared parameters, props, imported values, or type members.
         * Output: The declared return value, rendered elements, exported value, or type contract.
         */
        const variant = toast.variant ?? 'info'
        /**
         * What: This declaration defines the named behavior or data shape.
         * How: It processes the values supplied by callers according to its implementation.
         * Input: Declared parameters, props, imported values, or type members.
         * Output: The declared return value, rendered elements, exported value, or type contract.
         */
        const Icon = iconMap[variant]
        return (
          <div
            key={toast.id}
            className="pointer-events-auto animate-slide-in rounded-xl border border-brand-border bg-brand-card p-3 text-sm shadow-lg"
          >
            <div className="flex items-start gap-2">
              <Icon className="mt-0.5 h-4 w-4 text-brand-bright" />
              <div className="flex-1">
                <p className="font-semibold text-brand-text">{toast.title}</p>
                {toast.description ? <p className="text-brand-muted">{toast.description}</p> : null}
              </div>
              <button
                type="button"
                aria-label="Dismiss notification"
                onClick={() => dismissToast(toast.id)}
                className="rounded-md p-1 text-brand-muted transition hover:bg-brand-card2"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
