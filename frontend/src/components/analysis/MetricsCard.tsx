interface MetricsCardProps {
  label: string
  value: string
}

export const MetricsCard = ({ label, value }: MetricsCardProps) => (
  <div className="rounded-xl border border-brand-border bg-brand-card2/45 p-4">
    <p className="text-xs uppercase tracking-wide text-brand-muted">{label}</p>
    <p className="mt-1 text-xl font-semibold text-brand-text">{value}</p>
  </div>
)
