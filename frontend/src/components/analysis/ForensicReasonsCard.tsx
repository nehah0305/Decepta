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
    <div className="glass-effect rounded-2xl p-6 space-y-6 border border-brand-border/60">
      {/* Verdict Banner */}
      <div
        className={`rounded-2xl p-6 border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 ${
          isFake
            ? 'bg-red-500/10 border-red-500/30 text-red-200'
            : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
        }`}
      >
        <div className="flex items-center gap-4">
          <div
            className={`p-3.5 rounded-2xl ${
              isFake ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'
            }`}
          >
            {isFake ? <ShieldAlert className="h-8 w-8" /> : <CheckCircle2 className="h-8 w-8" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-widest font-semibold opacity-80">
                Classification Verdict
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-orbitron font-bold mt-0.5">
              {isFake ? '🔴 DEEPFAKE DETECTED (FAKE)' : '🟢 MEDIA VERIFIED (GENUINE REAL)'}
            </h2>
            <p className="text-sm opacity-90 mt-1">
              {isFake
                ? `Model is ${confidence}% confident this media contains artificial facial or spatial manipulations.`
                : `Model is ${(100 - confidence).toFixed(
                    1
                  )}% confident this media contains natural organic facial structures.`}
            </p>
          </div>
        </div>

        <div className="shrink-0 px-5 py-2.5 rounded-xl glass-effect-sm border border-current text-right">
          <p className="text-xs uppercase tracking-wider opacity-75">Visual Confidence</p>
          <p className="text-2xl font-orbitron font-bold">{confidence}%</p>
        </div>
      </div>

      {/* Forensic Reasons & Locations Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2 text-brand-bright">
            <Layers className="h-5 w-5" />
            <h3 className="text-lg font-orbitron font-bold text-brand-text">
              Forensic Analysis & Anomaly Localization
            </h3>
          </div>
          <span className="text-xs text-brand-subtle">
            ResNet-50 Fine-Tuned Spatial & Spectral Engine
          </span>
        </div>

        {reasons.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-3">
            {reasons.map((item, idx) => (
              <div
                key={idx}
                className="rounded-xl border border-brand-border bg-brand-card/50 p-5 space-y-3 hover:border-brand-bright/40 transition-colors flex flex-col justify-between"
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-brand-bright">
                      Reason #{idx + 1}
                    </span>
                    <span
                      className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${
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

                  <h4 className="text-base font-semibold text-brand-text leading-tight">
                    {item.category}
                  </h4>

                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-brand-card2/80 text-brand-bright text-xs border border-brand-border">
                    <Crosshair className="h-3.5 w-3.5 shrink-0" />
                    <span className="font-mono text-[11px] truncate">{item.location}</span>
                  </div>

                  <p className="text-xs text-brand-subtle leading-relaxed mt-2">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-brand-border bg-brand-card/30 p-5 text-sm text-brand-subtle">
            <p>Analysis completed. No anomalous spatial-frequency artifacts detected.</p>
          </div>
        )}
      </div>
    </div>
  )
}
