import type { ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  fullWidth?: boolean
  leftIcon?: ReactNode
}

const variants: Record<Variant, string> = {
  primary:
    'bg-gradient-to-r from-brand-primary to-brand-bright text-brand-bg shadow-[0_0_24px_rgba(80,214,209,0.35)] hover:brightness-110',
  secondary: 'bg-brand-card2 text-brand-text border border-brand-border hover:bg-brand-card',
  ghost: 'bg-transparent text-brand-subtle hover:bg-brand-card/60',
  danger: 'bg-rose-500/20 text-rose-200 border border-rose-400/40 hover:bg-rose-500/30',
}

export const Button = ({
  children,
  variant = 'primary',
  fullWidth,
  leftIcon,
  className = '',
  ...props
}: ButtonProps) => (
  <button
    className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${fullWidth ? 'w-full' : ''} ${className}`}
    {...props}
  >
    {leftIcon}
    {children}
  </button>
)
