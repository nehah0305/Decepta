interface ProgressProps {
  value: number
}

export const Progress = ({ value }: ProgressProps) => (
  <div className="h-2 w-full overflow-hidden rounded-full bg-brand-bg/80">
    <div
      className="h-full rounded-full bg-gradient-to-r from-brand-primary to-brand-bright transition-all duration-500"
      style={{ width: `${value}%` }}
    ></div>
  </div>
)
