// What: This file implements the useToast module in frontend/src/hooks/useToast.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { generateUniqueId } from '../utils/id'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface ToastItem {
  id: string
  title: string
  description?: string
  variant?: 'success' | 'error' | 'info'
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface ToastContextValue {
  toasts: ToastItem[]
  showToast: (toast: Omit<ToastItem, 'id'>) => void
  dismissToast: (id: string) => void
}

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const ToastContext = createContext<ToastContextValue | null>(null)

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const ToastProvider = ({ children }: { children: ReactNode }) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [toasts, setToasts] = useState<ToastItem[]>([])

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((item) => item.id !== id))
  }, [])

  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const showToast = useCallback(
    (toast: Omit<ToastItem, 'id'>) => {
      /**
       * What: This declaration defines the named behavior or data shape.
       * How: It processes the values supplied by callers according to its implementation.
       * Input: Declared parameters, props, imported values, or type members.
       * Output: The declared return value, rendered elements, exported value, or type contract.
       */
      const id = generateUniqueId('toast')
      setToasts((prev) => [...prev, { ...toast, id }])
      window.setTimeout(() => dismissToast(id), 3200)
    },
    [dismissToast],
  )

  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const value = useMemo(
    () => ({ toasts, showToast, dismissToast }),
    [dismissToast, showToast, toasts],
  )

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const useToast = () => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within ToastProvider')
  }

  return context
}
