import type { ReactNode } from 'react'

interface MetricsCardProps {
  label: string
  value: string
  icon?: ReactNode
}

export const MetricsCard = ({ label, value, icon }: MetricsCardProps) => (
  <div className="rounded-2xl border border-brand-border bg-brand-card/60 p-4 hover:border-brand-bright/40 transition-colors space-y-1.5 backdrop-blur-sm">
    <div className="flex items-center justify-between">
      <p className="text-[11px] font-bold uppercase tracking-wider text-brand-muted">{label}</p>
      {icon ? <span className="text-brand-bright">{icon}</span> : null}
    </div>
    <p className="text-lg font-bold font-orbitron text-brand-text truncate">{value}</p>
  </div>
)
