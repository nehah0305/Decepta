import type { FormEvent } from 'react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Card } from '../components/ui/Card'
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
    <Card className="border-brand-bright/30 bg-brand-surface/70 p-6 sm:p-8">
      <h1 className="text-2xl font-semibold text-brand-text">Login</h1>
      <form className="mt-5 space-y-4" onSubmit={submit} noValidate>
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

        <div className="text-right">
          <a className="text-xs text-brand-bright hover:underline" href="#" aria-label="Forgot password link">
            Forgot password?
          </a>
        </div>

        <Button type="submit" fullWidth disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </Button>
      </form>
      <p className="mt-4 text-center text-sm text-brand-muted">
        Don&apos;t have an account?{' '}
        <Link to="/register" className="text-brand-bright hover:underline">
          Register
        </Link>
      </p>
    </Card>
  )
}
