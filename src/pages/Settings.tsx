import { useEffect, useState } from 'react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Input } from '../components/ui/Input'
import { Toggle } from '../components/ui/Toggle'
import { DEFAULT_SETTINGS, STORAGE_KEYS } from '../data/mockData'
import { useToast } from '../hooks/useToast'
import { SettingsState } from '../types'

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
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-3xl font-semibold text-brand-text">Settings</h1>
      </header>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-brand-text">General</h2>
          <Input
            label="Application Name"
            value={settings.appName}
            onChange={(event) => setSettings((prev) => ({ ...prev, appName: event.target.value }))}
          />
          <label className="space-y-1.5">
            <span className="text-sm text-brand-subtle">Theme</span>
            <select
              className="w-full rounded-xl border border-brand-border bg-brand-card2/70 px-3 py-2.5 text-sm text-brand-subtle"
              value={settings.theme}
              onChange={(event) =>
                setSettings((prev) => ({ ...prev, theme: event.target.value as SettingsState['theme'] }))
              }
            >
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </label>
          <label className="space-y-1.5">
            <span className="text-sm text-brand-subtle">Language</span>
            <select
              className="w-full rounded-xl border border-brand-border bg-brand-card2/70 px-3 py-2.5 text-sm text-brand-subtle"
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
        </Card>

        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-brand-text">Detection</h2>
          <label className="space-y-1.5">
            <span className="text-sm text-brand-subtle">Default Detection Type</span>
            <select
              className="w-full rounded-xl border border-brand-border bg-brand-card2/70 px-3 py-2.5 text-sm text-brand-subtle"
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
            label="Confidence Threshold"
            type="number"
            min={0}
            max={100}
            value={String(settings.confidenceThreshold)}
            onChange={(event) =>
              setSettings((prev) => ({ ...prev, confidenceThreshold: Number(event.target.value) }))
            }
          />
          <Toggle
            label="Automatic processing"
            checked={settings.autoProcessing}
            onChange={(checked) => setSettings((prev) => ({ ...prev, autoProcessing: checked }))}
          />
        </Card>

        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-brand-text">Notifications</h2>
          <Toggle
            label="Email notifications"
            checked={settings.emailNotifications}
            onChange={(checked) => setSettings((prev) => ({ ...prev, emailNotifications: checked }))}
          />
          <Toggle
            label="Detection completion notifications"
            checked={settings.completionNotifications}
            onChange={(checked) => setSettings((prev) => ({ ...prev, completionNotifications: checked }))}
          />
        </Card>

        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-brand-text">Security</h2>
          <Input label="Change Password" type="password" placeholder="••••••••" />
          <Button variant="secondary">Session management</Button>
        </Card>
      </div>

      <Button type="button" onClick={save} disabled={saving}>
        {saving ? 'Saving...' : 'Save Changes'}
      </Button>
    </div>
  )
}
