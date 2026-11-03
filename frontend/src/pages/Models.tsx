// What: This file implements the Models module in frontend/src/pages/Models.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { useState } from 'react'
import { Sparkles, Activity, Database, Volume2, Video } from 'lucide-react'
import { Badge } from '../components/ui/Badge'
import { AblationTable } from '../components/analysis/AblationTable'

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const Models = () => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [activeTab, setActiveTab] = useState<'spatial' | 'fft' | 'temporal' | 'acoustic'>('spatial')

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright font-semibold">
          <Sparkles className="h-4 w-4" />
          <span>Research & Multi-Branch Benchmarks</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Model Architecture Explorer</h1>
        <p className="text-brand-subtle text-lg">
          Inspect trained model weights, ablation benchmarks, dataset coverage, and feature representation maps.
        </p>
      </header>

      {/* Dual Champion Model Banners Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Visual Champion Model */}
        <div className="glass-effect rounded-3xl p-7 border border-brand-bright/40 bg-gradient-to-br from-brand-surface via-brand-card to-brand-surface space-y-5 shadow-2xl relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3 relative z-10">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-brand-bright/15 border border-brand-bright/30 text-brand-bright text-xs font-bold uppercase tracking-widest">
                <Video className="h-3.5 w-3.5" /> Visual Champion
              </span>
              <span className="text-xs font-mono text-brand-muted">DFD 30 Epochs</span>
            </div>
            <h2 className="text-2xl font-orbitron font-bold text-brand-text">
              ResNet-50 Stage B (Visual)
            </h2>
            <p className="text-brand-subtle text-xs leading-relaxed">
              Fine-tuned deep convolutional layers (layer3 + layer4) across 30 epochs on 2,917 DeepFakeDetection (DFD) video clips. Pretrained ImageNet representations isolate face boundary blending warping and facial seam discontinuities.
            </p>
          </div>

          <div className="space-y-4 pt-2">
            <div className="grid grid-cols-4 gap-2 text-center">
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">ROC-AUC</p>
                <p className="text-lg font-bold font-orbitron text-brand-bright">95.03%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Precision</p>
                <p className="text-lg font-bold font-orbitron text-brand-text">99.7%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">F1-Score</p>
                <p className="text-lg font-bold font-orbitron text-purple-400">94.2%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Params</p>
                <p className="text-lg font-bold font-orbitron text-brand-text">23.5M</p>
              </div>
            </div>
          </div>
        </div>

        {/* Audio Champion Model */}
        <div className="glass-effect rounded-3xl p-7 border border-cyan-500/40 bg-gradient-to-br from-brand-surface via-brand-card to-brand-surface space-y-5 shadow-2xl relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3 relative z-10">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 text-xs font-bold uppercase tracking-widest">
                <Volume2 className="h-3.5 w-3.5" /> Audio Champion
              </span>
              <span className="text-xs font-mono text-brand-muted">LAV-DF Test Split</span>
            </div>
            <h2 className="text-2xl font-orbitron font-bold text-brand-text">
              Acoustic Transformer (Audio)
            </h2>
            <p className="text-brand-subtle text-xs leading-relaxed">
              Trained 2-layer Transformer encoder with attention pooling over 68 human-interpretable acoustic descriptors (20 MFCCs, Δ, Δ², F0 pitch, spectral rolloff, ZCR, energy) from <code className="text-cyan-300">audio/audio-deepfake-detect</code>.
            </p>
          </div>

          <div className="space-y-4 pt-2">
            <div className="grid grid-cols-4 gap-2 text-center">
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">ROC-AUC</p>
                <p className="text-lg font-bold font-orbitron text-cyan-400">97.23%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Precision</p>
                <p className="text-lg font-bold font-orbitron text-brand-text">97.2%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Accuracy</p>
                <p className="text-lg font-bold font-orbitron text-emerald-400">91.4%</p>
              </div>
              <div className="glass-effect-sm rounded-xl p-2.5 border border-brand-border">
                <p className="text-[10px] uppercase font-bold text-brand-muted">Params</p>
                <p className="text-lg font-bold font-orbitron text-brand-text">286K</p>
              </div>
            </div>
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

          <div className="flex items-center gap-1.5 bg-brand-card/60 p-1.5 rounded-2xl border border-brand-border flex-wrap">
            <button
              type="button"
              onClick={() => setActiveTab('spatial')}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'spatial'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              Spatial Grad-CAM
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('fft')}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'fft'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              2D FFT Spectrum
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('temporal')}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'temporal'
                  ? 'bg-brand-bright text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              Temporal Motion
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('acoustic')}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                activeTab === 'acoustic'
                  ? 'bg-cyan-400 text-brand-bg shadow-md'
                  : 'text-brand-subtle hover:text-brand-text'
              }`}
            >
              Acoustic SHAP
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
                  : activeTab === 'temporal'
                  ? 'Inter-Frame Temporal Variance Map'
                  : 'KernelSHAP Acoustic Descriptor Attribution'
              }
              tone="neutral"
            />
            <h4 className="text-xl font-bold text-brand-text font-orbitron">
              {activeTab === 'spatial' && 'Edge & Facial Contour Gradient Heatmap'}
              {activeTab === 'fft' && 'High-Frequency Neural Upsampling Grid Artifacts'}
              {activeTab === 'temporal' && 'Temporal Frame-to-Frame Jitter Vector Field'}
              {activeTab === 'acoustic' && '68-Descriptor Acoustic Cepstral & Formant SHAP'}
            </h4>
            <p className="text-sm text-brand-subtle leading-relaxed">
              {activeTab === 'spatial' &&
                'Highlights localized spatial gradient anomalies where face-swapping algorithms blend synthetic inner face regions with original neck and hairline borders.'}
              {activeTab === 'fft' &&
                'Exposes high-frequency grid noise produced by transposed convolution layers during GAN/Diffusion generator upsampling.'}
              {activeTab === 'temporal' &&
                'Tracks micro-fluctuations in facial landmark velocity across video frame sequences, detecting temporal boundary flickering.'}
              {activeTab === 'acoustic' &&
                'Maps KernelSHAP attributions across 20 MFCCs, velocity deltas, acceleration deltas, F0 fundamental pitch contour, and spectral rolloff to attribute synthesis to specific vocoder filters.'}
            </p>

            <div className="rounded-2xl glass-effect-sm p-4 border border-brand-border space-y-2">
              <p className="text-xs uppercase font-bold text-brand-bright tracking-wider">Domain Signal Rating</p>
              <div className="flex items-center justify-between text-xs text-brand-text font-semibold">
                <span>Signal Reliability</span>
                <span className="text-emerald-400 font-orbitron font-bold">
                  {activeTab === 'acoustic' ? 'EXCELLENT (0.972 ROC-AUC)' : 'EXCELLENT (0.950 ROC-AUC)'}
                </span>
              </div>
            </div>
          </div>

          {/* Visual Display Mockup */}
          <div className="rounded-2xl border-2 border-brand-bright/40 bg-brand-bg p-6 text-center space-y-4 relative overflow-hidden group">
            <div className="h-64 rounded-xl bg-gradient-to-br from-[#143239] via-[#0b242c] to-[#142633] flex flex-col items-center justify-center space-y-3 border border-brand-border p-4">
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
              {activeTab === 'acoustic' && (
                <div className="w-full space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-mono text-cyan-300">
                    <span>mfcc_delta_12</span>
                    <span className="text-red-400 font-bold">+3.684 (High Risk)</span>
                  </div>
                  <div className="h-2 w-full bg-cyan-950 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-amber-400 to-red-500 rounded-full w-[85%]"></div>
                  </div>

                  <div className="flex items-center justify-between text-[11px] font-mono text-cyan-300">
                    <span>mfcc_16</span>
                    <span className="text-red-400 font-bold">+2.652 (High Risk)</span>
                  </div>
                  <div className="h-2 w-full bg-cyan-950 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-amber-400 to-red-500 rounded-full w-[65%]"></div>
                  </div>

                  <div className="flex items-center justify-between text-[11px] font-mono text-cyan-300">
                    <span>spectral_rolloff</span>
                    <span className="text-amber-300 font-bold">+1.645 (Vocoder Cutoff)</span>
                  </div>
                  <div className="h-2 w-full bg-cyan-950 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-teal-400 to-amber-400 rounded-full w-[45%]"></div>
                  </div>
                  <p className="text-[10px] text-cyan-400 font-mono pt-1">
                    KernelSHAP Attributions • 68 Descriptors
                  </p>
                </div>
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
            <h3 className="text-xl font-orbitron font-bold text-brand-text">DeepFakeDetection (DFD)</h3>
          </div>
          <p className="text-sm text-brand-subtle leading-relaxed">
            The primary visual benchmark dataset for ResNet-50 Stage B training. Features 3,431 high-coverage video sequences (2,398 train, 519 validation, 514 test) with 100% face extraction and mixed-precision acceleration.
          </p>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-brand-border/40 font-semibold">
            <span className="text-brand-muted">Visual Training Status</span>
            <span className="text-brand-bright font-bold">30 Epochs Complete (95.03% ROC-AUC)</span>
          </div>
        </div>

        <div className="glass-effect rounded-3xl p-8 space-y-4 border border-cyan-500/40">
          <div className="flex items-center gap-3 text-cyan-400">
            <Volume2 className="h-6 w-6" />
            <h3 className="text-xl font-orbitron font-bold text-brand-text">LAV-DF (Audio Benchmark)</h3>
          </div>
          <p className="text-sm text-brand-subtle leading-relaxed">
            Primary benchmark dataset for the Acoustic Transformer in <code className="text-cyan-300">audio/audio-deepfake-detect</code>. Evaluated across thousands of speech samples with 68 named acoustic descriptors and KernelSHAP explainability.
          </p>
          <div className="flex items-center justify-between text-xs pt-2 border-t border-brand-border/40 font-semibold">
            <span className="text-brand-muted">Acoustic Status</span>
            <span className="text-cyan-400 font-bold">Tested & Deployed (97.23% ROC-AUC)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
