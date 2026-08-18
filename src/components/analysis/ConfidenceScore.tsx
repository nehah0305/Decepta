import { DetectionRecord } from '../../types'

export const ConfidenceScore = ({ confidence }: Pick<DetectionRecord, 'confidence'>) => (
  <div className="relative h-32 w-32">
    <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
      <circle cx="60" cy="60" r="48" stroke="rgba(130,220,225,0.18)" strokeWidth="10" fill="none" />
      <circle
        cx="60"
        cy="60"
        r="48"
        stroke="#50D6D1"
        strokeWidth="10"
        fill="none"
        strokeDasharray={301.59}
        strokeDashoffset={301.59 - (301.59 * confidence) / 100}
        strokeLinecap="round"
      />
    </svg>
    <div className="absolute inset-0 grid place-items-center text-center">
      <p className="text-2xl font-semibold text-brand-text">{confidence}%</p>
      <p className="text-xs uppercase tracking-wide text-brand-muted">Confidence</p>
    </div>
  </div>
)
