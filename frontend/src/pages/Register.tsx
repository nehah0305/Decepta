import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Sparkles, ArrowRight, ShieldCheck } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../hooks/useToast'

export const Register = () => {
  const { register } = useAuth()
  const navigate = useNavigate()
  const { showToast } = useToast()
  const [form, setForm] = useState({ name: '', email: '', password: '', confirmPassword: '' })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)

  const submit = async (event: FormEvent) => {
    event.preventDefault()

    const nextErrors: Record<string, string> = {}

    if (!form.name.trim()) nextErrors.name = 'Name is required.'
    if (!form.email.trim()) nextErrors.email = 'Email is required.'
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) nextErrors.email = 'Enter a valid email.'
    if (form.password.length < 8) nextErrors.password = 'Password must be at least 8 characters.'
    if (form.password !== form.confirmPassword)
      nextErrors.confirmPassword = 'Passwords do not match.'

    setErrors(nextErrors)
    if (Object.keys(nextErrors).length) return

    setLoading(true)
    await register({ name: form.name, email: form.email, password: form.password })
    setLoading(false)

    showToast({ title: 'Account created', description: 'Your profile is ready.', variant: 'success' })
    navigate('/detect')
  }

  return (
    <div className="w-full space-y-6 max-w-md mx-auto">
      {/* Header */}
      <div className="space-y-2 text-center">
        <div className="flex justify-center mb-2">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-bright via-brand-primary to-[#142633] text-brand-bg shadow-lg glow-effect">
            <ShieldCheck className="h-7 w-7 text-brand-bg" />
          </div>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-3.5 w-3.5" />
          <span>DECEPTA Suite Registration</span>
        </div>
        <h1 className="text-3xl font-orbitron font-bold gradient-text pt-1">Create Account</h1>
        <p className="text-brand-subtle text-sm">Join DECEPTA to detect deepfakes & verify media authenticity</p>
      </div>

      {/* Register Form Card */}
      <div className="glass-effect rounded-3xl px-8 py-8 space-y-6 animate-fade-in border border-brand-border/70 shadow-2xl">
        <form className="space-y-4" onSubmit={submit} noValidate>
          <Input
            label="Full Name"
            value={form.name}
            onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
            error={errors.name}
            required
          />
          <Input
            label="Email"
            type="email"
            value={form.email}
            onChange={(event) => setForm((prev) => ({ ...prev, email: event.target.value }))}
            error={errors.email}
            required
          />
          <Input
            label="Password"
            type="password"
            value={form.password}
            onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
            error={errors.password}
            required
          />
          <Input
            label="Confirm Password"
            type="password"
            value={form.confirmPassword}
            onChange={(event) => setForm((prev) => ({ ...prev, confirmPassword: event.target.value }))}
            error={errors.confirmPassword}
            required
          />

          <Button type="submit" fullWidth disabled={loading} className="glow-effect py-3.5 font-bold text-base rounded-2xl">
            {loading ? 'Creating Account...' : 'Create Account'}
            {!loading && <ArrowRight className="h-4 w-4 ml-1" />}
          </Button>
        </form>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-brand-border/40"></div>
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="bg-brand-surface px-3 text-brand-muted font-medium">or</span>
          </div>
        </div>

        {/* Sign In Link */}
        <p className="text-center text-sm text-brand-muted">
          Already have an account?{' '}
          <Link to="/login" className="text-brand-bright hover:underline font-bold transition-colors">
            Login here
          </Link>
        </p>
      </div>
    </div>
  )
}
