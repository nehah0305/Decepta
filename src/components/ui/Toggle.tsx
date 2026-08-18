interface ToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}

export const Toggle = ({ label, checked, onChange }: ToggleProps) => (
  <label className="flex items-center justify-between gap-4 rounded-xl border border-brand-border bg-brand-card2/40 px-3 py-2.5">
    <span className="text-sm text-brand-subtle">{label}</span>
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative h-6 w-11 rounded-full border transition ${checked ? 'border-brand-bright bg-brand-primary/50' : 'border-brand-border bg-brand-bg/60'}`}
    >
      <span
        className={`absolute top-0.5 h-[18px] w-[18px] rounded-full bg-brand-text transition ${checked ? 'left-5' : 'left-0.5'}`}
      ></span>
    </button>
  </label>
)
