import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { STORAGE_KEYS } from '../data/mockData'
import type { AuthUser } from '../types'

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
  const [user, setUser] = useState<AuthUser | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEYS.auth)
    if (stored) {
      setUser(JSON.parse(stored) as AuthUser)
    }
  }, [])

  const persist = (nextUser: AuthUser | null) => {
    if (nextUser) {
      localStorage.setItem(STORAGE_KEYS.auth, JSON.stringify(nextUser))
      localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(nextUser))
    } else {
      localStorage.removeItem(STORAGE_KEYS.auth)
    }
  }

  const register = async ({ name, email }: RegisterInput) => {
    await wait(700)
    const nextUser: AuthUser = {
      id: crypto.randomUUID(),
      name,
      email,
      username: name.toLowerCase().replace(/\s+/g, '_'),
    }
    setUser(nextUser)
    persist(nextUser)
  }

  const login = async ({ email }: LoginInput) => {
    await wait(500)
    const profile = localStorage.getItem(STORAGE_KEYS.profile)
    const nextUser = profile
      ? (JSON.parse(profile) as AuthUser)
      : {
          id: crypto.randomUUID(),
          name: 'Alex Rivera',
          email,
          username: 'alex_rivera',
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
