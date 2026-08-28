import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Sparkles, ArrowRight } from 'lucide-react'
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
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="space-y-2 text-center">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright animate-pulse-glow mb-4">
          <Sparkles className="h-4 w-4" />
          <span>Depecta</span>
        </div>
        <h1 className="text-3xl font-orbitron font-bold text-brand-text">Create Account</h1>
        <p className="text-brand-subtle">Join Depecta to detect deepfakes</p>
      </div>

      {/* Register Form Card */}
      <div className="glass-effect rounded-2xl px-8 py-8 space-y-6 animate-fade-in">
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

          <Button type="submit" fullWidth disabled={loading} className="glow-effect py-3 font-semibold">
            {loading ? 'Creating Account...' : 'Create Account'}
            {!loading && <ArrowRight className="h-4 w-4" />}
          </Button>
        </form>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-brand-border/30"></div>
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="bg-gradient-to-b from-transparent via-brand-bg to-transparent px-2 text-brand-muted">or</span>
          </div>
        </div>

        {/* Sign In Link */}
        <p className="text-center text-sm text-brand-muted">
          Already have an account?{' '}
          <Link to="/login" className="text-brand-bright hover:text-brand-bright/80 font-semibold transition-colors">
            Login here
          </Link>
        </p>
      </div>
    </div>
  )
}