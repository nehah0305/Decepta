import { createContext, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { STORAGE_KEYS, getSafeStorageItem } from '../data/mockData'
import type { AuthUser } from '../types'
import { generateUniqueId } from '../utils/id'

interface RegisterInput {
  name: string
  email: string
  password: string
}

interface LoginInput {
  email: string
  password: string
}

interface AuthContextValue {
  user: AuthUser | null
  isAuthenticated: boolean
  login: (input: LoginInput) => Promise<void>
  register: (input: RegisterInput) => Promise<void>
  logout: () => void
  updateProfile: (input: Partial<AuthUser>) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  // Synchronous lazy state initialization from localStorage - prevents refresh redirects
  const [user, setUser] = useState<AuthUser | null>(() => {
    try {
      const stored = getSafeStorageItem('auth')
      return stored ? (JSON.parse(stored) as AuthUser) : null
    } catch {
      return null
    }
  })

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

  const register = async ({ name, email }: RegisterInput) => {
    await wait(700)
    const nextUser: AuthUser = {
      id: generateUniqueId('usr'),
      name,
      email,
      username: name.toLowerCase().replace(/\s+/g, '_'),
    }
    setUser(nextUser)
    persist(nextUser)
  }

  const login = async ({ email }: LoginInput) => {
    await wait(500)
    let nextUser: AuthUser
    try {
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

  const updateProfile = (input: Partial<AuthUser>) => {
    setUser((current) => {
      if (!current) {
        return null
      }
      const nextUser = { ...current, ...input }
      persist(nextUser)
      return nextUser
    })
  }

  const logout = () => {
    setUser(null)
    persist(null)
  }

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

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider')
  }

  return context
}
