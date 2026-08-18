import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
}

export const Card = ({ children, className = '' }: CardProps) => (
  <section
    className={`rounded-2xl border border-brand-border bg-brand-card/80 p-5 shadow-[0_20px_48px_rgba(5,18,22,0.55)] backdrop-blur-sm ${className}`}
  >
    {children}
  </section>
)
