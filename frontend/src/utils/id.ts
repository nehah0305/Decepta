// What: This file implements the id module in frontend/src/utils/id.ts.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

/**
 * Robust unique ID generator with safe fallback for non-secure / legacy environments
 */
export const generateUniqueId = (prefix = 'id'): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    try {
      return crypto.randomUUID()
    } catch {
      // Fallback
    }
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 9)}`
}

