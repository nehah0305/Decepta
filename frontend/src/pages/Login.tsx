import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Sparkles, ArrowRight } from 'lucide-react'
import { Input } from '../components/ui/Input'
import { Button } from '../components/ui/Button'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../hooks/useToast'

export const Login = () => {
  const navigate = useNavigate()
  const { login } = useAuth()
  const { showToast } = useToast()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)

  const submit = async (event: FormEvent) => {
    event.preventDefault()

    const nextErrors: Record<string, string> = {}
    if (!email) nextErrors.email = 'Email is required.'
    if (!password) nextErrors.password = 'Password is required.'

    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return

    setLoading(true)
    await login({ email, password })
    setLoading(false)

    showToast({ title: 'Welcome back', description: 'Authentication successful.', variant: 'success' })
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
        <h1 className="text-3xl font-orbitron font-bold text-brand-text">Welcome Back</h1>
        <p className="text-brand-subtle">Login to continue to Depecta</p>
      </div>

      {/* Login Form Card */}
      <div className="glass-effect rounded-2xl px-8 py-8 space-y-6 animate-fade-in">
        <form className="space-y-4" onSubmit={submit} noValidate>
          <Input
            label="Email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            error={errors.email}
            required
          />
          <Input
            label="Password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            error={errors.password}
            required
          />

          <div className="flex justify-end">
            <a className="text-xs text-brand-bright hover:text-brand-bright/80 transition-colors" href="#" aria-label="Forgot password link">
              Forgot password?
            </a>
          </div>

          <Button type="submit" fullWidth disabled={loading} className="glow-effect py-3 font-semibold">
            {loading ? 'Logging in...' : 'Login'}
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

        {/* Sign Up Link */}
        <p className="text-center text-sm text-brand-muted">
          Don&apos;t have an account?{' '}
          <Link to="/register" className="text-brand-bright hover:text-brand-bright/80 font-semibold transition-colors">
            Create one
          </Link>
        </p>
      </div>
    </div>
  )
}
