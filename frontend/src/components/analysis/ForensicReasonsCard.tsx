// What: This file implements the ForensicReasonsCard module in frontend/src/components/analysis/ForensicReasonsCard.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { CheckCircle2, Crosshair, Layers, ShieldAlert, Volume2, Activity } from 'lucide-react'
import type { AcousticShapItem, DetectionType, ForensicReason } from '../../types'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface ForensicReasonsCardProps {
  verdict?: 'DEEPFAKE' | 'GENUINE'
  resultLabel: string
  confidence: number
  fileType?: DetectionType
  reasons?: ForensicReason[]
  acousticShap?: AcousticShapItem[]
}

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const ForensicReasonsCard = ({
  verdict = 'DEEPFAKE',
  resultLabel,
  confidence,
  fileType = 'video',
  reasons = [],
  acousticShap = [],
}: ForensicReasonsCardProps) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const isFake = verdict === 'DEEPFAKE' || resultLabel.includes('FAKE')
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const isAudio = fileType === 'audio'

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
                {isAudio ? 'Acoustic Stream Verdict' : 'Spatial-Visual Verdict'}
              </span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-orbitron font-extrabold tracking-tight truncate">
              {isFake
                ? isAudio
                  ? '🔴 SYNTHETIC VOICE DETECTED (FAKE)'
                  : '🔴 DEEPFAKE DETECTED (FAKE)'
                : isAudio
                ? '🟢 GENUINE SPEECH VERIFIED (REAL)'
                : '🟢 MEDIA VERIFIED (GENUINE REAL)'}
            </h2>
            <p className="text-xs sm:text-sm opacity-90 leading-relaxed max-w-2xl">
              {isFake
                ? isAudio
                  ? `Model is ${confidence}% confident this audio track was artificially generated or cloned using neural vocoders / TTS.`
                  : `Model is ${confidence}% confident this media contains artificial facial or spatial manipulations.`
                : isAudio
                ? `Model is ${(100 - confidence).toFixed(
                    1
                  )}% confident this audio contains natural human vocal tract acoustics and breath harmonics.`
                : `Model is ${(100 - confidence).toFixed(
                    1
                  )}% confident this media contains natural organic facial structures.`}
            </p>
          </div>
        </div>

        <div className="shrink-0 w-full sm:w-auto px-6 py-3 rounded-2xl glass-effect-sm border border-current text-left sm:text-right">
          <p className="text-[10px] uppercase font-bold tracking-wider opacity-80">
            {isAudio ? 'Acoustic Confidence' : 'Visual Confidence'}
          </p>
          <p className="text-3xl font-orbitron font-extrabold">{confidence}%</p>
        </div>
      </div>

      {/* Acoustic SHAP Attributions Bar (for Audio Stream Audit) */}
      {isAudio && acousticShap.length > 0 && (
        <div className="rounded-2xl border border-cyan-500/40 bg-cyan-950/20 p-5 space-y-4">
          <div className="flex items-center justify-between gap-2 flex-wrap border-b border-cyan-500/20 pb-3">
            <div className="flex items-center gap-2 text-cyan-300">
              <Activity className="h-5 w-5" />
              <h4 className="font-orbitron font-bold text-sm uppercase tracking-wider">
                KernelSHAP Acoustic Feature Attribution (68-Descriptor Audit)
              </h4>
            </div>
            <span className="text-[11px] font-mono text-cyan-400">
              (+ pushes toward synthetic, - pushes toward genuine)
            </span>
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            {acousticShap.map((item, i) => {
              /**
               * What: This declaration defines the named behavior or data shape.
               * How: It processes the values supplied by callers according to its implementation.
               * Input: Declared parameters, props, imported values, or type members.
               * Output: The declared return value, rendered elements, exported value, or type contract.
               */
              const isPositive = item.value >= 0
              /**
               * What: This declaration defines the named behavior or data shape.
               * How: It processes the values supplied by callers according to its implementation.
               * Input: Declared parameters, props, imported values, or type members.
               * Output: The declared return value, rendered elements, exported value, or type contract.
               */
              const absVal = Math.min(Math.abs(item.value) / 4.0, 1.0)
              /**
               * What: This declaration defines the named behavior or data shape.
               * How: It processes the values supplied by callers according to its implementation.
               * Input: Declared parameters, props, imported values, or type members.
               * Output: The declared return value, rendered elements, exported value, or type contract.
               */
              const widthPct = Math.max(absVal * 100, 8)

              return (
                <div
                  key={i}
                  className="flex items-center justify-between gap-3 p-2.5 rounded-xl bg-brand-card/60 border border-brand-border/60 text-xs"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[11px] font-semibold text-brand-text truncate">
                        {item.feature}
                      </span>
                      <span
                        className={`font-mono font-bold text-[11px] ${
                          isPositive ? 'text-red-400' : 'text-emerald-400'
                        }`}
                      >
                        {isPositive ? `+${item.value.toFixed(3)}` : item.value.toFixed(3)}
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-brand-card2 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isPositive ? 'bg-gradient-to-r from-amber-500 to-red-500' : 'bg-gradient-to-r from-teal-500 to-emerald-500'
                        }`}
                        style={{ width: `${widthPct}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Forensic Reasons & Anomaly Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2 flex-wrap border-b border-brand-border/40 pb-3">
          <div className="flex items-center gap-2.5 text-brand-bright">
            {isAudio ? <Volume2 className="h-5 w-5" /> : <Layers className="h-5 w-5" />}
            <h3 className="text-lg font-orbitron font-bold text-brand-text">
              {isAudio ? 'Acoustic Forensic Anomaly Localization' : 'Forensic Anomaly Breakdown & Localization'}
            </h3>
          </div>
          <span className="text-xs text-brand-subtle font-mono">
            {isAudio ? 'Acoustic Transformer (LAV-DF Model)' : 'ResNet-50 Fine-Tuned Spatial & 2D FFT Engine'}
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
            <p className="text-xs">
              {isAudio
                ? 'Acoustic formant harmonics and 68 descriptors indicate organic vocal tract physiology.'
                : 'Spatial and 2D FFT spectral transforms indicate organic facial coherence.'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
