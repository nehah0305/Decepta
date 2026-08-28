import type { DetectionRecord } from '../../types'

export const ConfidenceScore = ({ confidence }: Pick<DetectionRecord, 'confidence'>) => (
  <div className="relative h-36 w-36 flex items-center justify-center">
    <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
      <circle cx="60" cy="60" r="48" stroke="rgba(130,220,225,0.15)" strokeWidth="8" fill="none" />
      <circle
        cx="60"
        cy="60"
        r="48"
        stroke="url(#confidenceGradient)"
        strokeWidth="9"
        fill="none"
        strokeDasharray={301.59}
        strokeDashoffset={301.59 - (301.59 * confidence) / 100}
        strokeLinecap="round"
        className="transition-all duration-1000 ease-out"
      />
      <defs>
        <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#50d6d1" />
          <stop offset="50%" stopColor="#26b8b5" />
          <stop offset="100%" stopColor="#38bdf8" />
        </linearGradient>
      </defs>
    </svg>
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-2">
      <p className="text-3xl font-orbitron font-extrabold text-brand-bright drop-shadow-md">{confidence}%</p>
      <p className="text-[10px] uppercase font-bold tracking-widest text-brand-muted mt-0.5">Confidence</p>
    </div>
  </div>
)
