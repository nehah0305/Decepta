import type { TimelineEvent } from '../../types'

interface DetectionTimelineProps {
  events: TimelineEvent[]
}

export const DetectionTimeline = ({ events }: DetectionTimelineProps) => (
  <div className="space-y-4">
    {/* Bar Visualizer */}
    <div className="h-3.5 overflow-hidden rounded-full bg-brand-bg/80 border border-brand-border/60 p-0.5 flex items-center">
      {events.map((event) => (
        <div
          key={event.label}
          className={`h-full rounded-full transition-all duration-500 ${
            event.active ? 'bg-gradient-to-r from-brand-primary to-brand-bright glow-effect-sm' : 'bg-brand-card2/60'
          }`}
          style={{ width: `${event.end - event.start}%` }}
          title={`${event.label}: ${event.start}s – ${event.end}s`}
        ></div>
      ))}
    </div>

    {/* Stage Cards */}
    <div className="grid gap-2.5 sm:grid-cols-2 md:grid-cols-4">
      {events.map((event) => (
        <div
          key={event.label}
          className={`rounded-xl border p-3 space-y-1 transition-all ${
            event.active
              ? 'border-brand-bright/40 bg-brand-bright/10 text-brand-bright'
              : 'border-brand-border/50 bg-brand-card/40 text-brand-subtle'
          }`}
        >
          <div className="flex items-center justify-between text-xs font-semibold">
            <span className="truncate">{event.label}</span>
            <span className="font-mono text-[10px] opacity-80">{event.end - event.start}%</span>
          </div>
          <p className="text-[11px] font-mono text-brand-muted">
            {event.start}s – {event.end}s
          </p>
        </div>
      ))}
    </div>
  </div>
)
