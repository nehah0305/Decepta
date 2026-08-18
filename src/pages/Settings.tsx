import { useEffect, useState } from 'react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Toggle } from '../components/ui/Toggle'
import { DEFAULT_SETTINGS, STORAGE_KEYS } from '../data/mockData'
import { useToast } from '../hooks/useToast'
import type { SettingsState } from '../types'
import { Sparkles, Save } from 'lucide-react'

export const Settings = () => {
  const { showToast } = useToast()
  const [saving, setSaving] = useState(false)
  const [settings, setSettings] = useState<SettingsState>(DEFAULT_SETTINGS)

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEYS.settings)
    if (stored) {
      setSettings(JSON.parse(stored) as SettingsState)
    }
  }, [])

  const save = async () => {
    setSaving(true)
    await new Promise((resolve) => window.setTimeout(resolve, 600))
    localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(settings))
    setSaving(false)
    showToast({ title: 'Settings saved', variant: 'success' })
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-4 w-4" />
          <span>Configuration</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Settings</h1>
        <p className="text-brand-subtle text-lg">Customize your detection preferences and account settings.</p>
      </header>

      {/* Settings Grid */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* General Settings */}
        <div className="glass-effect rounded-2xl p-8 space-y-6 border border-brand-border/50">
          <div className="space-y-2">
            <h2 className="text-2xl font-semibold text-brand-text">General</h2>
            <p className="text-sm text-brand-subtle">Basic application preferences</p>
          </div>
          
          <div className="space-y-4">
            <Input
              label="Application Name"
              value={settings.appName}
              onChange={(event) => setSettings((prev) => ({ ...prev, appName: event.target.value }))}
            />
            <label className="space-y-2">
              <span className="text-sm font-medium text-brand-text">Theme</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card2/40 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
                value={settings.theme}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, theme: event.target.value as SettingsState['theme'] }))
                }
              >
                <option value="dark">Dark</option>
                <option value="system">System</option>
              </select>
            </label>
            <label className="space-y-2">
              <span className="text-sm font-medium text-brand-text">Language</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card2/40 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
                value={settings.language}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, language: event.target.value as SettingsState['language'] }))
                }
              >
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="de">German</option>
              </select>
            </label>
          </div>
        </div>

        {/* Detection Settings */}
        <div className="glass-effect rounded-2xl p-8 space-y-6 border border-brand-border/50">
          <div className="space-y-2">
            <h2 className="text-2xl font-semibold text-brand-text">Detection</h2>
            <p className="text-sm text-brand-subtle">AI detection preferences</p>
          </div>
          
          <div className="space-y-4">
            <label className="space-y-2">
              <span className="text-sm font-medium text-brand-text">Default Detection Type</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card2/40 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
                value={settings.defaultDetectionType}
                onChange={(event) =>
                  setSettings((prev) => ({
                    ...prev,
                    defaultDetectionType: event.target.value as SettingsState['defaultDetectionType'],
                  }))
                }
              >
                <option value="video">Video</option>
                <option value="audio">Audio</option>
              </select>
            </label>
            <Input
              label="Confidence Threshold (%)"
              type="number"
              min={0}
              max={100}
              value={String(settings.confidenceThreshold)}
              onChange={(event) =>
                setSettings((prev) => ({ ...prev, confidenceThreshold: Number(event.target.value) }))
              }
            />
            <Toggle
              label="Automatic Processing"
              checked={settings.autoProcessing}
              onChange={(checked) => setSettings((prev) => ({ ...prev, autoProcessing: checked }))}
            />
          </div>
        </div>

        {/* Notifications */}
        <div className="glass-effect rounded-2xl p-8 space-y-6 border border-brand-border/50">
          <div className="space-y-2">
            <h2 className="text-2xl font-semibold text-brand-text">Notifications</h2>
            <p className="text-sm text-brand-subtle">Alert preferences</p>
          </div>
          
          <div className="space-y-4">
            <Toggle
              label="Email Notifications"
              checked={settings.emailNotifications}
              onChange={(checked) => setSettings((prev) => ({ ...prev, emailNotifications: checked }))}
            />
            <Toggle
              label="Detection Completion Alerts"
              checked={settings.completionNotifications}
              onChange={(checked) => setSettings((prev) => ({ ...prev, completionNotifications: checked }))}
            />
          </div>
        </div>

        {/* Security */}
        <div className="glass-effect rounded-2xl p-8 space-y-6 border border-brand-border/50">
          <div className="space-y-2">
            <h2 className="text-2xl font-semibold text-brand-text">Security</h2>
            <p className="text-sm text-brand-subtle">Protect your account</p>
          </div>
          
          <div className="space-y-4">
            <Input label="Change Password" type="password" placeholder="••••••••" />
            <Button variant="secondary" fullWidth>Manage Sessions</Button>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button type="button" onClick={save} disabled={saving} className="glow-effect px-8 py-3 font-semibold text-base" leftIcon={<Save className="h-5 w-5" />}>
          {saving ? 'Saving...' : 'Save All Changes'}
        </Button>
      </div>
    </div>
  )
}
