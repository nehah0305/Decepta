import { Award, Layers, ShieldCheck, Sparkles, Volume2, Video } from 'lucide-react'

export const AblationTable = () => {
  const ablationData = [
    {
      model: 'Spatial CNN (Scratch)',
      modality: 'Visual',
      roc: '52.62%',
      notes: 'Baseline spatial architecture',
    },
    {
      model: 'ResNet-50 (Frozen Backbone)',
      modality: 'Visual',
      roc: '56.82%',
      notes: 'Feature extraction only',
    },
    {
      model: 'FFT 2D Frequency CNN',
      modality: 'Spectral',
      roc: '58.44%',
      notes: '2D FFT Log-Magnitude domain',
    },
    {
      model: 'Spatial + FFT Concat Fusion',
      modality: 'Visual+FFT',
      roc: '61.49%',
      notes: 'Spatial + 2D FFT concatenation',
    },
    {
      model: 'Spatial + Temporal Transformer',
      modality: 'Visual+Time',
      roc: '64.54%',
      notes: 'Sequence-level modeling',
    },
    {
      model: 'ResNet-50 (Stage B, FF++ Baseline)',
      modality: 'Visual',
      roc: '72.88%',
      notes: 'Initial visual baseline on FF++',
    },
    {
      model: 'ResNet-50 Stage B (Full 30-Epoch DFD)',
      modality: 'Visual',
      roc: '95.03%',
      notes: '⭐ Visual Branch Champion (99.70% Precision, 94.23% F1 on 3,431 DFD clips)',
      highlight: true,
      badge: 'Visual Champion',
    },
    {
      model: 'Acoustic Transformer (LAV-DF Audio Branch)',
      modality: 'Audio',
      roc: '97.23%',
      notes: '⭐ Audio Branch Champion (68 Descriptors, 97.25% Precision, 286K params)',
      highlight: true,
      badge: 'Audio Champion',
    },
    {
      model: 'Bidirectional Cross-Attention Fusion (LAV-DF)',
      modality: 'Multimodal',
      roc: '99.32%',
      notes: '⭐ Joint Audio-Visual Champion (99.55% Precision, 4.9M params)',
      highlight: true,
      badge: 'Fusion Champion',
    },
  ]

  return (
    <div className="glass-effect rounded-2xl p-6 space-y-6 border border-brand-border/60">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-3 py-1 text-xs font-semibold text-brand-bright uppercase tracking-widest mb-2">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Experimental Benchmark Matrix</span>
          </div>
          <h3 className="text-xl font-orbitron font-bold text-brand-text">
            Multi-Branch Forensic Ablation Study
          </h3>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-brand-bright/10 border border-brand-bright/30 text-brand-bright text-xs font-medium">
            <Video className="h-3.5 w-3.5" />
            <span>Visual: 95.03% AUC</span>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-medium">
            <Volume2 className="h-3.5 w-3.5" />
            <span>Audio: 97.23% AUC</span>
          </div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-medium">
            <Award className="h-3.5 w-3.5" />
            <span>Fusion: 99.32% AUC</span>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-brand-border bg-brand-card/40">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-brand-border bg-brand-card2/60 text-xs uppercase tracking-wider text-brand-muted">
            <tr>
              <th className="py-3.5 px-4 font-semibold">Model Architecture</th>
              <th className="py-3.5 px-3 font-semibold text-center">Modality</th>
              <th className="py-3.5 px-4 font-semibold text-center">Validation ROC-AUC</th>
              <th className="py-3.5 px-4 font-semibold">Research Findings & Deployment Status</th>
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
                <td className="py-3 px-3 text-center text-xs font-mono text-brand-subtle">
                  {row.modality}
                </td>
                <td className="py-3 px-4 text-center font-orbitron font-bold text-base">
                  {row.roc}
                </td>
                <td className="py-3 px-4 text-xs text-brand-subtle">
                  <div className="flex items-center gap-2">
                    {row.badge && (
                      <span className="px-2 py-0.5 rounded-full bg-brand-bright/20 text-brand-bright text-[10px] font-bold shrink-0">
                        {row.badge}
                      </span>
                    )}
                    <span>{row.notes}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-xl glass-effect-sm p-4 border border-brand-border/60 space-y-3">
        <div className="flex items-center gap-2 text-brand-bright text-xs font-semibold uppercase tracking-wider">
          <Layers className="h-4 w-4" />
          <span>Research Conclusion & Multi-Domain Architecture</span>
        </div>
        <p className="text-xs leading-relaxed text-brand-subtle">
          Rather than treating deepfake detection as a single monolithic classifier, Decepta pairs a fine-tuned spatial ResNet-50 visual backbone (<strong className="text-brand-bright">95.03% ROC-AUC</strong> on DFD) with a dedicated 68-descriptor Acoustic Transformer (<strong className="text-cyan-400">97.23% ROC-AUC</strong>, <strong className="text-emerald-400">97.25% Precision</strong> on LAV-DF). Both branches are independently validated and provide explainable forensic attributions via Spatial Grad-CAM and Acoustic KernelSHAP.
        </p>
      </div>

      <div className="rounded-xl bg-emerald-500/10 p-4 border border-emerald-500/30 text-emerald-300 text-xs space-y-1">
        <div className="flex items-center gap-2 font-semibold text-emerald-200">
          <ShieldCheck className="h-4 w-4" />
          <span>Integrated Audio Pipeline Status</span>
        </div>
        <p className="leading-relaxed">
          The acoustic detection pipeline from <code className="bg-emerald-950/60 px-1.5 py-0.5 rounded text-emerald-200">audio/audio-deepfake-detect</code> is fully integrated with 286,410 trained parameters, 68 named acoustic descriptors, and KernelSHAP explainability. Audio stream audits now directly leverage this dedicated acoustic model.
        </p>
      </div>
    </div>
  )
}
