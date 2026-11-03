// What: This file implements the useAuth module in frontend/src/hooks/useAuth.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { createContext, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { STORAGE_KEYS, getSafeStorageItem } from '../data/mockData'
import type { AuthUser } from '../types'
import { generateUniqueId } from '../utils/id'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface RegisterInput {
  name: string
  email: string
  password: string
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface LoginInput {
  email: string
  password: string
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface AuthContextValue {
  user: AuthUser | null
  isAuthenticated: boolean
  login: (input: LoginInput) => Promise<void>
  register: (input: RegisterInput) => Promise<void>
  logout: () => void
  updateProfile: (input: Partial<AuthUser>) => void
}

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const AuthContext = createContext<AuthContextValue | null>(null)

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const AuthProvider = ({ children }: { children: ReactNode }) => {
  // Synchronous lazy state initialization from localStorage - prevents refresh redirects
  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [user, setUser] = useState<AuthUser | null>(() => {
    try {
      /**
       * What: This declaration defines the named behavior or data shape.
       * How: It processes the values supplied by callers according to its implementation.
       * Input: Declared parameters, props, imported values, or type members.
       * Output: The declared return value, rendered elements, exported value, or type contract.
       */
      const stored = getSafeStorageItem('auth')
      return stored ? (JSON.parse(stored) as AuthUser) : null
    } catch {
      return null
    }
  })

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const persist = (nextUser: AuthUser | null) => {
    try {
      if (nextUser) {
        localStorage.setItem(STORAGE_KEYS.auth, JSON.stringify(nextUser))
        localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(nextUser))
      } else {
        localStorage.removeItem(STORAGE_KEYS.auth)
      }
    } catch {
      // ignore
    }
  }

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const register = async ({ name, email }: RegisterInput) => {
    await wait(700)
    /**
     * What: This declaration defines the named behavior or data shape.
     * How: It processes the values supplied by callers according to its implementation.
     * Input: Declared parameters, props, imported values, or type members.
     * Output: The declared return value, rendered elements, exported value, or type contract.
     */
    const nextUser: AuthUser = {
      id: generateUniqueId('usr'),
      name,
      email,
      username: name.toLowerCase().replace(/\s+/g, '_'),
    }
    setUser(nextUser)
    persist(nextUser)
  }

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const login = async ({ email }: LoginInput) => {
    await wait(500)
    let nextUser: AuthUser
    try {
      /**
       * What: This declaration defines the named behavior or data shape.
       * How: It processes the values supplied by callers according to its implementation.
       * Input: Declared parameters, props, imported values, or type members.
       * Output: The declared return value, rendered elements, exported value, or type contract.
       */
      const profile = getSafeStorageItem('profile')
      nextUser = profile
        ? (JSON.parse(profile) as AuthUser)
        : {
            id: generateUniqueId('usr'),
            name: 'Alex Rivera',
            email,
            username: 'alex_rivera',
          }
    } catch {
      nextUser = {
        id: generateUniqueId('usr'),
        name: 'Alex Rivera',
        email,
        username: 'alex_rivera',
      }
    }

    setUser(nextUser)
    persist(nextUser)
  }

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const updateProfile = (input: Partial<AuthUser>) => {
    setUser((current) => {
      if (!current) {
        return null
      }
      /**
       * What: This declaration defines the named behavior or data shape.
       * How: It processes the values supplied by callers according to its implementation.
       * Input: Declared parameters, props, imported values, or type members.
       * Output: The declared return value, rendered elements, exported value, or type contract.
       */
      const nextUser = { ...current, ...input }
      persist(nextUser)
      return nextUser
    })
  }

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const logout = () => {
    setUser(null)
    persist(null)
  }

  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const value = useMemo(
    () => ({
      user,
      isAuthenticated: Boolean(user),
      login,
      register,
      logout,
      updateProfile,
    }),
    [user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const useAuth = () => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider')
  }

  return context
}
