import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Input } from '../components/ui/Input'
import { STORAGE_KEYS } from '../data/mockData'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../hooks/useToast'
import type { AuthUser } from '../types'

export const User = () => {
  const navigate = useNavigate()
  const { user, logout, updateProfile } = useAuth()
  const { showToast } = useToast()
  const [saving, setSaving] = useState(false)
  const [profile, setProfile] = useState<AuthUser | null>(user)

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEYS.profile)
    if (stored) {
      setProfile(JSON.parse(stored) as AuthUser)
    }
  }, [])

  if (!profile) {
    return null
  }

  const save = async () => {
    setSaving(true)
    await new Promise((resolve) => window.setTimeout(resolve, 600))
    localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(profile))
    updateProfile(profile)
    setSaving(false)
    showToast({ title: 'Profile updated', variant: 'success' })
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-3xl font-semibold text-brand-text">User</h1>
      </header>

      <Card>
        <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
          <div className="flex items-center justify-center">
            <div className="grid h-40 w-40 place-items-center rounded-full border border-brand-border bg-brand-card2/40 text-4xl font-semibold text-brand-bright">
              {profile.name.slice(0, 1).toUpperCase()}
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <Input
                label="Full Name"
                value={profile.name}
                onChange={(event) => setProfile((prev) => (prev ? { ...prev, name: event.target.value } : prev))}
              />
              <Input
                label="Email"
                type="email"
                value={profile.email}
                onChange={(event) =>
                  setProfile((prev) => (prev ? { ...prev, email: event.target.value } : prev))
                }
              />
            </div>
            <Input
              label="Username"
              value={profile.username}
              onChange={(event) =>
                setProfile((prev) => (prev ? { ...prev, username: event.target.value } : prev))
              }
            />
            <div className="flex flex-wrap gap-3">
              <Button type="button" onClick={save} disabled={saving}>
                {saving ? 'Saving...' : 'Save Changes'}
              </Button>
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  logout()
                  navigate('/login')
                }}
              >
                Logout
              </Button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
