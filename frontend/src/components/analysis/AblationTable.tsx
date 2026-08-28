import { Award, Layers, ShieldAlert, Sparkles } from 'lucide-react'

export const AblationTable = () => {
  const ablationData = [
    { model: 'Spatial CNN (Scratch)', roc: '52.62%', notes: 'Baseline spatial architecture' },
    { model: 'ResNet-50 (Frozen Backbone)', roc: '56.82%', notes: 'Feature extraction only' },
    { model: 'FFT 2D Frequency CNN', roc: '58.44%', notes: '2D FFT Log-Magnitude domain' },
    { model: 'Spatial + FFT Concat Fusion', roc: '61.49%', notes: 'Spatial + 2D FFT concatenation' },
    { model: 'Spatial + Temporal Transformer', roc: '64.54%', notes: 'Sequence-level modeling' },
    { model: 'ResNet-50 (Fine-Tuned Stage B)', roc: '72.88%', notes: '⭐ Best Validated Visual Model', highlight: true },
  ]

  return (
    <div className="glass-effect rounded-2xl p-6 space-y-6 border border-brand-border/60">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-3 py-1 text-xs font-semibold text-brand-bright uppercase tracking-widest mb-2">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Experimental Benchmark</span>
          </div>
          <h3 className="text-xl font-orbitron font-bold text-brand-text">Visual Branch Ablation Study</h3>
        </div>
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-brand-bright/10 border border-brand-bright/30 text-brand-bright text-xs font-medium">
          <Award className="h-4 w-4" />
          <span>Validated on 320 FF++ Samples</span>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-brand-border bg-brand-card/40">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-brand-border bg-brand-card2/60 text-xs uppercase tracking-wider text-brand-muted">
            <tr>
              <th className="py-3.5 px-4 font-semibold">Model Architecture</th>
              <th className="py-3.5 px-4 font-semibold text-center">Validation ROC-AUC</th>
              <th className="py-3.5 px-4 font-semibold">Research Findings</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-brand-border/50 text-brand-text">
            {ablationData.map((row, idx) => (
              <tr
                key={idx}
                className={
                  row.highlight
                    ? 'bg-brand-bright/10 font-medium text-brand-bright'
                    : 'hover:bg-brand-card2/30 transition-colors'
                }
              >
                <td className="py-3 px-4 flex items-center gap-2">
                  {row.highlight && <Sparkles className="h-4 w-4 text-brand-bright shrink-0" />}
                  <span>{row.model}</span>
                </td>
                <td className="py-3 px-4 text-center font-orbitron font-bold text-base">
                  {row.roc}
                </td>
                <td className="py-3 px-4 text-xs text-brand-subtle">{row.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-xl glass-effect-sm p-4 border border-brand-border/60 space-y-3">
        <div className="flex items-center gap-2 text-brand-bright text-xs font-semibold uppercase tracking-wider">
          <Layers className="h-4 w-4" />
          <span>Research Conclusion & Key Insights</span>
        </div>
        <p className="text-xs leading-relaxed text-brand-subtle">
          Rather than assuming that adding more modalities automatically improves detection performance, we conducted controlled ablation experiments. Interestingly, naïve spatial-frequency fusion degraded performance. Fine-tuned spatial transfer learning produced the strongest validated visual baseline at <strong className="text-brand-bright">72.88% ROC-AUC</strong>.
        </p>
      </div>

      <div className="rounded-xl bg-amber-500/10 p-4 border border-amber-500/30 text-amber-300 text-xs space-y-1">
        <div className="flex items-center gap-2 font-semibold text-amber-200">
          <ShieldAlert className="h-4 w-4" />
          <span>Audio Pipeline Integrity Audit Note</span>
        </div>
        <p className="leading-relaxed">
          Following our mandatory audio integrity audit, we identified that standard FaceForensics++ video clips contain no valid audio tracks (100% silent). To ensure experimental validity, audio & sync pretraining have been halted on FF++ and will be retrained on an audio-capable dataset (FakeAVCeleb).
        </p>
      </div>
    </div>
  )
}
