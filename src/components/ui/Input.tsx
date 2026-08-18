import type { InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string
  error?: string
}

export const Input = ({ label, id, error, className = '', ...props }: InputProps) => {
  const inputId = id ?? label.toLowerCase().replace(/\s+/g, '-')

  return (
    <div className="space-y-1.5">
      <label htmlFor={inputId} className="text-sm text-brand-subtle">
        {label}
      </label>
      <input
        id={inputId}
        className={`w-full rounded-xl border border-brand-border bg-brand-card2/70 px-3.5 py-2.5 text-sm text-brand-text placeholder:text-brand-muted transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright ${error ? 'border-rose-400/50' : ''} ${className}`}
        {...props}
      />
      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  )
}
