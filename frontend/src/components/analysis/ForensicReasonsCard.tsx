import { CheckCircle2, Crosshair, Layers, ShieldAlert } from 'lucide-react'
import type { ForensicReason } from '../../types'

interface ForensicReasonsCardProps {
  verdict?: 'DEEPFAKE' | 'GENUINE'
  resultLabel: string
  confidence: number
  reasons?: ForensicReason[]
}

export const ForensicReasonsCard = ({
  verdict = 'DEEPFAKE',
  resultLabel,
  confidence,
  reasons = [],
}: ForensicReasonsCardProps) => {
  const isFake = verdict === 'DEEPFAKE' || resultLabel.includes('FAKE') || confidence > 50

  return (
    <div className="glass-effect rounded-3xl p-6 sm:p-8 space-y-6 border border-brand-border/70 shadow-2xl">
      {/* Verdict Banner */}
      <div
        className={`rounded-2xl p-6 border flex flex-col md:flex-row items-start md:items-center justify-between gap-6 transition-all ${
          isFake
            ? 'bg-red-500/10 border-red-500/30 text-red-200 shadow-[0_0_30px_rgba(239,68,68,0.15)]'
            : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200 shadow-[0_0_30px_rgba(16,185,129,0.15)]'
        }`}
      >
        <div className="flex items-start sm:items-center gap-4 min-w-0">
          <div
            className={`p-4 rounded-2xl shrink-0 ${
              isFake ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
            }`}
          >
            {isFake ? <ShieldAlert className="h-8 w-8" /> : <CheckCircle2 className="h-8 w-8" />}
          </div>
          <div className="space-y-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[11px] uppercase tracking-widest font-bold opacity-80">
                Classification Verdict
              </span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-orbitron font-extrabold tracking-tight truncate">
              {isFake ? '🔴 DEEPFAKE DETECTED (FAKE)' : '🟢 MEDIA VERIFIED (GENUINE REAL)'}
            </h2>
            <p className="text-xs sm:text-sm opacity-90 leading-relaxed max-w-2xl">
              {isFake
                ? `Model is ${confidence}% confident this media contains artificial facial or spatial manipulations.`
                : `Model is ${(100 - confidence).toFixed(
                    1
                  )}% confident this media contains natural organic facial structures.`}
            </p>
          </div>
        </div>

        <div className="shrink-0 w-full sm:w-auto px-6 py-3 rounded-2xl glass-effect-sm border border-current text-left sm:text-right">
          <p className="text-[10px] uppercase font-bold tracking-wider opacity-80">Visual Confidence</p>
          <p className="text-3xl font-orbitron font-extrabold">{confidence}%</p>
        </div>
      </div>

      {/* Forensic Reasons & Anomaly Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2 flex-wrap border-b border-brand-border/40 pb-3">
          <div className="flex items-center gap-2.5 text-brand-bright">
            <Layers className="h-5 w-5" />
            <h3 className="text-lg font-orbitron font-bold text-brand-text">
              Forensic Anomaly Breakdown & Localization
            </h3>
          </div>
          <span className="text-xs text-brand-subtle font-mono">
            ResNet-50 Fine-Tuned Spatial & 2D FFT Engine
          </span>
        </div>

        {reasons.length > 0 ? (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            {reasons.map((item, idx) => (
              <div
                key={idx}
                className="rounded-2xl border border-brand-border bg-brand-card/60 p-5 space-y-3.5 hover:border-brand-bright/50 transition-all flex flex-col justify-between backdrop-blur-sm"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-brand-bright font-orbitron">
                      Anomaly #{idx + 1}
                    </span>
                    <span
                      className={`text-[10px] uppercase font-bold px-2.5 py-0.5 rounded-full ${
                        item.severity === 'High'
                          ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                          : item.severity === 'Medium'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}
                    >
                      {item.severity} Risk
                    </span>
                  </div>

                  <h4 className="text-base font-bold text-brand-text leading-snug">
                    {item.category}
                  </h4>

                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-brand-card2/80 text-brand-bright text-xs border border-brand-border/60">
                    <Crosshair className="h-3.5 w-3.5 shrink-0 text-brand-bright" />
                    <span className="font-mono text-[11px] truncate">{item.location}</span>
                  </div>

                  <p className="text-xs text-brand-subtle leading-relaxed pt-1">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-2xl border border-brand-border bg-brand-card/40 p-6 text-sm text-brand-subtle text-center space-y-1">
            <p className="font-semibold text-brand-text">No Anomaly Artifacts Identified</p>
            <p className="text-xs">Spatial and 2D FFT spectral transforms indicate organic facial coherence.</p>
          </div>
        )}
      </div>
    </div>
  )
}
