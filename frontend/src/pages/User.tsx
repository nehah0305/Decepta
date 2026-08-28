import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { STORAGE_KEYS } from '../data/mockData'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../hooks/useToast'
import type { AuthUser } from '../types'
import { Sparkles, LogOut, Save, User as UserIcon, Award } from 'lucide-react'

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
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright font-semibold">
          <Sparkles className="h-4 w-4" />
          <span>User Profile & Credentials</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Researcher Profile</h1>
        <p className="text-brand-subtle text-lg">Manage your account credentials, security clearance, and activity logs.</p>
      </header>

      {/* Profile Card */}
      <div className="glass-effect rounded-3xl p-8 border border-brand-border/70 shadow-2xl space-y-8">
        <div className="grid gap-8 lg:grid-cols-[240px_1fr]">
          {/* Avatar & Role */}
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="relative">
              <div className="grid h-44 w-44 place-items-center rounded-3xl border-2 border-brand-bright/40 bg-gradient-to-br from-brand-bright/20 via-brand-primary/20 to-[#142633] text-6xl font-orbitron font-bold gradient-text shadow-2xl">
                {profile.name ? profile.name.slice(0, 1).toUpperCase() : 'U'}
              </div>
              <div className="absolute -bottom-2 -right-2 h-14 w-14 rounded-2xl bg-brand-card border-2 border-brand-bright grid place-items-center backdrop-blur-md shadow-lg">
                <UserIcon className="h-7 w-7 text-brand-bright" />
              </div>
            </div>

            <div className="space-y-1">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-brand-bright/15 border border-brand-bright/30 text-brand-bright text-xs font-bold uppercase tracking-wider">
                <Award className="h-3.5 w-3.5" /> Lead Researcher
              </span>
              <p className="text-xs text-brand-muted font-mono pt-1">ID: RESEARCHER-8842</p>
            </div>
          </div>

          {/* Form */}
          <div className="space-y-6">
            <div>
              <p className="text-xs font-bold text-brand-muted uppercase tracking-widest mb-4">Account Information</p>
              <div className="grid gap-4 sm:grid-cols-2">
                <Input
                  label="Full Name"
                  value={profile.name}
                  onChange={(event) => setProfile((prev) => (prev ? { ...prev, name: event.target.value } : prev))}
                />
                <Input
                  label="Email Address"
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

            {/* Quick Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2">
              <div className="glass-effect-sm rounded-xl p-3.5 border border-brand-border space-y-0.5">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Audits Executed</p>
                <p className="text-lg font-bold font-orbitron text-brand-bright">24 Scans</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-3.5 border border-brand-border space-y-0.5">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Organization</p>
                <p className="text-sm font-bold text-brand-text truncate">DECEPTA Labs</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-3.5 border border-brand-border space-y-0.5 col-span-2 sm:col-span-1">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Security Clearance</p>
                <p className="text-sm font-bold text-emerald-400">Level 4 (Admin)</p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap gap-4 pt-4 border-t border-brand-border/40">
              <Button type="button" onClick={save} disabled={saving} className="glow-effect px-7 py-3 font-bold rounded-2xl" leftIcon={<Save className="h-5 w-5" />}>
                {saving ? 'Saving...' : 'Save Profile Changes'}
              </Button>
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  logout()
                  navigate('/login')
                }}
                className="px-6 py-3 font-semibold hover:bg-red-500/15 hover:text-red-400 transition-colors rounded-2xl"
                leftIcon={<LogOut className="h-5 w-5" />}
              >
                Sign Out
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
