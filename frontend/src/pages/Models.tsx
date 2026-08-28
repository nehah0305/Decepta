import { useState } from 'react'
import { Sparkles, Award, Activity, Database } from 'lucide-react'
import { Badge } from '../components/ui/Badge'
import { AblationTable } from '../components/analysis/AblationTable'

export const Models = () => {
  const [activeTab, setActiveTab] = useState<'spatial' | 'fft' | 'temporal'>('spatial')

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright font-semibold">
          <Sparkles className="h-4 w-4" />
          <span>Research & Experimental Benchmark</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Model Architecture Explorer</h1>
        <p className="text-brand-subtle text-lg">
          Inspect model weights, ablation benchmarks, dataset coverage, and feature representation maps.
        </p>
      </header>

      {/* Champion Model Banner */}
      <div className="glass-effect rounded-3xl p-8 border border-brand-bright/40 bg-gradient-to-br from-brand-surface via-brand-card to-brand-surface space-y-6 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-brand-bright/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-xl bg-brand-bright/15 border border-brand-bright/30 text-brand-bright text-xs font-bold uppercase tracking-widest">
              <Award className="h-4 w-4" /> Champion Visual Detector
            </div>
            <h2 className="text-3xl font-orbitron font-bold text-brand-text">
              ResNet-50 Fine-Tuned (Stage B)
            </h2>
            <p className="text-brand-subtle text-sm max-w-2xl leading-relaxed">
              Fine-tuned deep convolutional layers on 320 balanced FaceForensics++ validation videos. Pretrained ImageNet weights provide robust lower-level edge representation, while fine-tuning captures high-frequency facial border warping.
            </p>
          </div>

          <div className="shrink-0 glass-effect-sm rounded-2xl p-6 border border-brand-bright/40 text-center space-y-1">
            <p className="text-xs font-bold uppercase tracking-widest text-brand-muted">Validation ROC-AUC</p>
            <p className="text-4xl font-orbitron font-extrabold text-brand-bright">72.88%</p>
            <p className="text-[11px] text-emerald-400 font-semibold">+20.26% over baseline</p>
          </div>
        </div>

        {/* Detailed Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-brand-border/50">
          <div className="glass-effect-sm rounded-xl p-4 border border-brand-border text-center">
            <p className="text-xs uppercase font-semibold text-brand-muted">Params</p>
            <p className="text-xl font-bold font-orbitron text-brand-text">23.5M</p>
          </div>
          <div className="glass-effect-sm rounded-xl p-4 border border-brand-border text-center">
            <p className="text-xs uppercase font-semibold text-brand-muted">Precision</p>
            <p className="text-xl font-bold font-orbitron text-brand-bright">71.4%</p>
          </div>
          <div className="glass-effect-sm rounded-xl p-4 border border-brand-border text-center">
            <p className="text-xs uppercase font-semibold text-brand-muted">Recall</p>
            <p className="text-xl font-bold font-orbitron text-emerald-400">73.8%</p>
          </div>
          <div className="glass-effect-sm rounded-xl p-4 border border-brand-border text-center">
            <p className="text-xs uppercase font-semibold text-brand-muted">F1-Score</p>
            <p className="text-xl font-bold font-orbitron text-purple-400">72.6%</p>
          </div>
        </div>
      </div>

      {/* Feature Map Visualizer Simulator */}
      <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border/70 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-brand-border/50 pb-4">
          <div>
            <h3 className="text-2xl font-orbitron font-bold text-brand-text">Feature Representation Explorer</h3>
            <p className="text-sm text-brand-subtle">Compare domain feature activations across modalities.</p>
          </div>

          <div className="flex items-center gap-2 bg-brand-card/60 p-1.5 rounded-2xl border border-brand-border">
            <button
              onClick={() => setActiveTab('spatial')}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'spatial'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              Spatial Grad-CAM
            </button>
            <button
              onClick={() => setActiveTab('fft')}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'fft'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              2D FFT Spectrum
            </button>
            <button
              onClick={() => setActiveTab('temporal')}
              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'temporal'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              Temporal Motion
            </button>
          </div>
        </div>

        {/* Simulator Preview Box */}
        <div className="grid gap-6 md:grid-cols-2 items-center">
          <div className="space-y-4">
            <Badge
              label={
                activeTab === 'spatial'
                  ? 'RGB Spatial Grad-CAM Overlay'
                  : activeTab === 'fft'
                  ? '2D FFT Log-Magnitude Frequency Plane'
                  : 'Inter-Frame Temporal Variance Map'
              }
              tone="neutral"
            />
            <h4 className="text-xl font-bold text-brand-text font-orbitron">
              {activeTab === 'spatial' && 'Edge & Facial Contour Gradient Heatmap'}
              {activeTab === 'fft' && 'High-Frequency Neural Upsampling Grid Artifacts'}
              {activeTab === 'temporal' && 'Temporal Frame-to-Frame Jitter Vector Field'}
            </h4>
            <p className="text-sm text-brand-subtle leading-relaxed">
              {activeTab === 'spatial' &&
                'Highlights localized spatial gradient anomalies where face-swapping algorithms blend synthetic inner face regions with original neck and hairline borders.'}
              {activeTab === 'fft' &&
                'Exposes high-frequency grid noise produced by transposed convolution layers during GAN/Diffusion generator upsampling.'}
              {activeTab === 'temporal' &&
                'Tracks micro-fluctuations in facial landmark velocity across video frame sequences, detecting temporal boundary flickering.'}
            </p>

            <div className="rounded-2xl glass-effect-sm p-4 border border-brand-border space-y-2">
              <p className="text-xs uppercase font-bold text-brand-bright tracking-wider">Domain Signal Rating</p>
              <div className="flex items-center justify-between text-xs text-brand-text font-semibold">
                <span>Signal Reliability</span>
                <span className="text-emerald-400 font-orbitron font-bold">HIGH (0.728 ROC-AUC)</span>
              </div>
            </div>
          </div>

          {/* Visual Display Mockup */}
          <div className="rounded-2xl border-2 border-brand-bright/40 bg-brand-bg p-6 text-center space-y-4 relative overflow-hidden group">
            <div className="h-64 rounded-xl bg-gradient-to-br from-[#143239] via-[#0b242c] to-[#142633] flex flex-col items-center justify-center space-y-3 border border-brand-border">
              {activeTab === 'spatial' && (
                <>
                  <div className="h-28 w-28 rounded-full border-4 border-dashed border-red-500/80 bg-red-500/20 animate-pulse flex items-center justify-center text-red-300 font-orbitron font-bold text-xs">
                    GRAD-CAM HOTSPOT
                  </div>
                  <p className="text-xs text-brand-bright font-mono">Face Boundary Anomaly (Jawline)</p>
                </>
              )}
              {activeTab === 'fft' && (
                <>
                  <div className="grid grid-cols-4 gap-2 p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/40">
                    {Array.from({ length: 8 }).map((_, i) => (
                      <div key={i} className="h-8 w-8 rounded bg-cyan-400/40 animate-pulse"></div>
                    ))}
                  </div>
                  <p className="text-xs text-cyan-300 font-mono">2D FFT High-Frequency Grid Noise</p>
                </>
              )}
              {activeTab === 'temporal' && (
                <>
                  <Activity className="h-16 w-16 text-purple-400 animate-pulse" />
                  <p className="text-xs text-purple-300 font-mono">Inter-Frame Landmark Jitter Spike</p>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Complete Ablation Table */}
      <AblationTable />

      {/* Dataset Audit Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="glass-effect rounded-3xl p-8 space-y-4 border border-brand-border/60">
          <div className="flex items-center gap-3 text-brand-bright">
            <Database className="h-6 w-6" />
            <h3 className="text-xl font-orbitron font-bold text-brand-text">FaceForensics++ (FF++)</h3>
          </div>
          <p className="text-sm text-brand-subtle leading-relaxed">
            The primary benchmark dataset utilized for visual branch fine-tuning and spatial ablation experiments. Contains 7,000 deepfake videos spanning Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
          </p>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-brand-border/40 font-semibold">
            <span className="text-brand-muted">Audio Coverage</span>
            <span className="text-amber-400">0% (All Tracks Silent)</span>
          </div>
        </div>

        <div className="glass-effect rounded-3xl p-8 space-y-4 border border-brand-border/60">
          <div className="flex items-center gap-3 text-emerald-400">
            <Database className="h-6 w-6" />
            <h3 className="text-xl font-orbitron font-bold text-brand-text">FakeAVCeleb</h3>
          </div>
          <p className="text-sm text-brand-subtle leading-relaxed">
            Target dataset designated for future multimodal expansion. Features 21,566 video clips with aligned real/synthesized audio and lip-sync modifications.
          </p>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-brand-border/40 font-semibold">
            <span className="text-brand-muted">Multimodal Status</span>
            <span className="text-emerald-400">Target Training Dataset</span>
          </div>
        </div>
      </div>
    </div>
  )
}
