interface BadgeProps {
  label: string
  tone?: 'success' | 'warning' | 'neutral'
}

const tones = {
  success: 'bg-emerald-500/20 text-emerald-200 border-emerald-300/35',
  warning: 'bg-amber-500/20 text-amber-200 border-amber-300/35',
  neutral: 'bg-brand-card2/60 text-brand-subtle border-brand-border',
}

export const Badge = ({ label, tone = 'neutral' }: BadgeProps) => (
  <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${tones[tone]}`}>
    {label}
  </span>
)
