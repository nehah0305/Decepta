// What: This file implements the Tabs module in frontend/src/components/ui/Tabs.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface TabItem<T extends string> {
  label: string
  value: T
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface TabsProps<T extends string> {
  value: T
  items: TabItem<T>[]
  onChange: (value: T) => void
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
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
