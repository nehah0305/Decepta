interface TabItem<T extends string> {
  label: string
  value: T
}

interface TabsProps<T extends string> {
  value: T
  items: TabItem<T>[]
  onChange: (value: T) => void
}

export const Tabs = <T extends string>({ value, items, onChange }: TabsProps<T>) => (
  <div className="inline-flex rounded-xl border border-brand-border bg-brand-card2/50 p-1">
    {items.map((item) => (
      <button
        key={item.value}
        type="button"
        onClick={() => onChange(item.value)}
        className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
          value === item.value
            ? 'bg-brand-primary/25 text-brand-bright'
            : 'text-brand-subtle hover:text-brand-text'
        }`}
      >
        {item.label}
      </button>
    ))}
  </div>
)
