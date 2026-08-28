import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Sparkles, ArrowRight, ShieldCheck } from 'lucide-react'
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
          <span>DECEPTA Suite Access</span>
        </div>
        <h1 className="text-3xl font-orbitron font-bold gradient-text pt-1">Welcome Back</h1>
        <p className="text-brand-subtle text-sm">Log in to access the deepfake forensic suite</p>
      </div>

      {/* Login Form Card */}
      <div className="glass-effect rounded-3xl px-8 py-8 space-y-6 animate-fade-in border border-brand-border/70 shadow-2xl">
        <form className="space-y-4" onSubmit={submit} noValidate>
          <Input
            label="Email"
            type="email"
            placeholder="researcher@decepta.ai"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            error={errors.email}
            required
          />
          <Input
            label="Password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            error={errors.password}
            required
          />

          <div className="flex justify-end">
            <a className="text-xs text-brand-bright hover:underline transition-colors" href="#" aria-label="Forgot password link">
              Forgot password?
            </a>
          </div>

          <Button type="submit" fullWidth disabled={loading} className="glow-effect py-3.5 font-bold rounded-2xl">
            {loading ? 'Authenticating...' : 'Sign In to Dashboard'}
            {!loading && <ArrowRight className="h-4 w-4 ml-1" />}
          </Button>
        </form>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-brand-border/40"></div>
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="bg-brand-surface px-3 text-brand-muted font-semibold uppercase">Or</span>
          </div>
        </div>

        {/* Sign Up Link */}
        <p className="text-center text-sm text-brand-muted">
          Don&apos;t have an account?{' '}
          <Link to="/register" className="text-brand-bright hover:underline font-bold transition-colors">
            Create an Account
          </Link>
        </p>
      </div>
    </div>
  )
}
