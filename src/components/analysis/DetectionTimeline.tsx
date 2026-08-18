import { TimelineEvent } from '../../types'

interface DetectionTimelineProps {
  events: TimelineEvent[]
}

export const DetectionTimeline = ({ events }: DetectionTimelineProps) => (
  <div>
    <div className="h-3 overflow-hidden rounded-full bg-brand-bg">
      {events.map((event) => (
        <span
          key={event.label}
          className={`inline-block h-full ${event.active ? 'bg-brand-bright' : 'bg-brand-card2'}`}
          style={{ width: `${event.end - event.start}%` }}
          title={event.label}
        ></span>
      ))}
    </div>
    <div className="mt-3 grid gap-2 sm:grid-cols-3">
      {events.map((event) => (
        <div key={event.label} className="rounded-lg border border-brand-border bg-brand-card2/35 px-3 py-2">
          <p className="text-xs text-brand-muted">{event.label}</p>
          <p className="text-sm text-brand-subtle">
            {event.start}s – {event.end}s
          </p>
        </div>
      ))}
    </div>
  </div>
)
