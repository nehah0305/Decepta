import { useEffect, useState } from 'react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Toggle } from '../components/ui/Toggle'
import { DEFAULT_SETTINGS, STORAGE_KEYS } from '../data/mockData'
import { useToast } from '../hooks/useToast'
import type { SettingsState } from '../types'
import { Sparkles, Save, Sliders, Bell, Shield, Monitor } from 'lucide-react'

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
    showToast({ title: 'System settings updated', variant: 'success' })
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright font-semibold">
          <Sparkles className="h-4 w-4" />
          <span>System Control & Preferences</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Settings</h1>
        <p className="text-brand-subtle text-lg">Configure your detection parameters, threshold triggers, and UI theme.</p>
      </header>

      {/* Settings Grid */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* General Settings */}
        <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border/60 shadow-xl">
          <div className="flex items-center gap-3 border-b border-brand-border/40 pb-4">
            <div className="p-2.5 rounded-xl bg-brand-bright/15 text-brand-bright">
              <Monitor className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-brand-text font-orbitron">Interface & Branding</h2>
              <p className="text-sm text-brand-subtle">General application preferences</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <Input
              label="Application Name"
              value={settings.appName}
              onChange={(event) => setSettings((prev) => ({ ...prev, appName: event.target.value }))}
            />
            <label className="space-y-2 block">
              <span className="text-sm font-semibold text-brand-text">Theme Mode</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card/60 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all font-medium"
                value={settings.theme}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, theme: event.target.value as SettingsState['theme'] }))
                }
              >
                <option value="dark">Cyber Dark Mode (Default)</option>
                <option value="system">System Default</option>
              </select>
            </label>
            <label className="space-y-2 block">
              <span className="text-sm font-semibold text-brand-text">Language</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card/60 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all font-medium"
                value={settings.language}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, language: event.target.value as SettingsState['language'] }))
                }
              >
                <option value="en">English (US)</option>
                <option value="es">Spanish</option>
                <option value="de">German</option>
              </select>
            </label>
          </div>
        </div>

        {/* Detection Settings */}
        <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border/60 shadow-xl">
          <div className="flex items-center gap-3 border-b border-brand-border/40 pb-4">
            <div className="p-2.5 rounded-xl bg-brand-bright/15 text-brand-bright">
              <Sliders className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-brand-text font-orbitron">AI Detection Engine</h2>
              <p className="text-sm text-brand-subtle">Precision and model threshold tuning</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <label className="space-y-2 block">
              <span className="text-sm font-semibold text-brand-text">Default Detection Domain</span>
              <select
                className="w-full rounded-xl border border-brand-border bg-brand-card/60 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all font-medium"
                value={settings.defaultDetectionType}
                onChange={(event) =>
                  setSettings((prev) => ({
                    ...prev,
                    defaultDetectionType: event.target.value as SettingsState['defaultDetectionType'],
                  }))
                }
              >
                <option value="video">Video Clips (ResNet-50 Fine-Tuned)</option>
                <option value="audio">Audio Streams (Spectrogram Audit)</option>
              </select>
            </label>
            <Input
              label="Confidence Sensitivity Threshold (%)"
              type="number"
              min={0}
              max={100}
              value={String(settings.confidenceThreshold)}
              onChange={(event) =>
                setSettings((prev) => ({ ...prev, confidenceThreshold: Number(event.target.value) }))
              }
            />
            <Toggle
              label="Automatic Pipeline Execution"
              checked={settings.autoProcessing}
              onChange={(checked) => setSettings((prev) => ({ ...prev, autoProcessing: checked }))}
            />
          </div>
        </div>

        {/* Notifications */}
        <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border/60 shadow-xl">
          <div className="flex items-center gap-3 border-b border-brand-border/40 pb-4">
            <div className="p-2.5 rounded-xl bg-brand-bright/15 text-brand-bright">
              <Bell className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-brand-text font-orbitron">Audit Alerts</h2>
              <p className="text-sm text-brand-subtle">Notification preferences & webhooks</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <Toggle
              label="Email Forensic Reports"
              checked={settings.emailNotifications}
              onChange={(checked) => setSettings((prev) => ({ ...prev, emailNotifications: checked }))}
            />
            <Toggle
              label="High-Risk Deepfake Alert Sound"
              checked={settings.completionNotifications}
              onChange={(checked) => setSettings((prev) => ({ ...prev, completionNotifications: checked }))}
            />
          </div>
        </div>

        {/* Security */}
        <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border/60 shadow-xl">
          <div className="flex items-center gap-3 border-b border-brand-border/40 pb-4">
            <div className="p-2.5 rounded-xl bg-brand-bright/15 text-brand-bright">
              <Shield className="h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-brand-text font-orbitron">Security & Access</h2>
              <p className="text-sm text-brand-subtle">Session keys and access credentials</p>
            </div>
          </div>
          
          <div className="space-y-4">
            <Input label="Master Password" type="password" placeholder="••••••••" />
            <Button variant="secondary" fullWidth className="py-3 font-semibold border-brand-border">
              Manage Active API Keys & Sessions
            </Button>
          </div>
        </div>
      </div>

      {/* Save Action */}
      <div className="flex justify-end pt-4">
        <Button type="button" onClick={save} disabled={saving} className="glow-effect px-8 py-3.5 font-bold text-base rounded-2xl" leftIcon={<Save className="h-5 w-5" />}>
          {saving ? 'Saving Preferences...' : 'Save All Preferences'}
        </Button>
      </div>
    </div>
  )
}
