import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { STORAGE_KEYS } from '../data/mockData'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../hooks/useToast'
import type { AuthUser } from '../types'
import { Sparkles, LogOut, Save, User as UserIcon } from 'lucide-react'

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
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-4 w-4" />
          <span>User Profile</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">My Account</h1>
        <p className="text-brand-subtle text-lg">Manage your profile information and preferences.</p>
      </header>

      {/* Profile Section */}
      <div className="glass-effect rounded-2xl p-8 border border-brand-border/50">
        <div className="grid gap-8 lg:grid-cols-[240px_1fr]">
          {/* Avatar */}
          <div className="flex items-center justify-center lg:justify-start">
            <div className="relative">
              <div className="grid h-48 w-48 place-items-center rounded-2xl border-2 border-brand-bright/30 bg-gradient-to-br from-brand-bright/20 to-brand-primary/20 text-6xl font-bold gradient-text shadow-xl">
                {profile.name.slice(0, 1).toUpperCase()}
              </div>
              <div className="absolute -bottom-2 -right-2 h-16 w-16 rounded-full bg-brand-bright/20 border-2 border-brand-bright grid place-items-center backdrop-blur-sm">
                <UserIcon className="h-8 w-8 text-brand-bright" />
              </div>
            </div>
          </div>

          {/* Form */}
          <div className="space-y-6">
            <div>
              <p className="text-sm font-medium text-brand-muted uppercase tracking-wider mb-4">Account Information</p>
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
              <div className="mt-4">
                <Input
                  label="Username"
                  value={profile.username}
                  onChange={(event) =>
                    setProfile((prev) => (prev ? { ...prev, username: event.target.value } : prev))
                  }
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap gap-4 pt-4 border-t border-brand-border/30">
              <Button type="button" onClick={save} disabled={saving} className="glow-effect px-6 py-3 font-semibold" leftIcon={<Save className="h-5 w-5" />}>
                {saving ? 'Saving...' : 'Save Changes'}
              </Button>
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  logout()
                  navigate('/login')
                }}
                className="px-6 py-3 font-semibold hover:bg-red-500/10 hover:text-red-400 transition-colors"
                leftIcon={<LogOut className="h-5 w-5" />}
              >
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
