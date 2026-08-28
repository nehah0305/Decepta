import { ArrowRight, ShieldCheck, Cpu, Layers, Activity, Lock, FileText } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'

export const Landing = () => (
  <div className="relative min-h-screen overflow-hidden text-brand-text bg-[#070B14]">
    {/* Ambient Glow Orbs */}
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute -top-40 -left-40 h-[500px] w-[500px] rounded-full bg-brand-bright/10 blur-[120px] animate-glow-drift"></div>
      <div className="absolute top-1/3 -right-40 h-[600px] w-[600px] rounded-full bg-brand-accent/10 blur-[150px] animate-glow-drift" style={{ animationDelay: '3s' }}></div>
      <div className="absolute -bottom-40 left-1/3 h-[500px] w-[500px] rounded-full bg-brand-neon/10 blur-[130px] animate-glow-drift" style={{ animationDelay: '6s' }}></div>
    </div>

    {/* Cyber Grid Overlay */}
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-30"></div>

    {/* Main Container */}
    <div className="relative z-10 mx-auto max-w-7xl px-6 pt-12 pb-24 space-y-24">
      {/* Top Brand Navigation */}
      <header className="flex items-center justify-between border-b border-brand-border/40 pb-6">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-bright to-brand-accent text-white shadow-lg glow-effect">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h2 className="font-orbitron text-2xl font-bold tracking-wider gradient-text">DECEPTA</h2>
            <p className="text-[10px] uppercase font-bold tracking-widest text-brand-muted">Deepfake Detection Suite</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <Link to="/login">
            <Button variant="secondary" className="px-5 py-2.5 text-sm font-semibold border-brand-border">
              Sign In
            </Button>
          </Link>
          <Link to="/detect">
            <Button className="px-5 py-2.5 text-sm font-semibold glow-effect">
              Launch Detector <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="text-center space-y-8 max-w-4xl mx-auto pt-6">
        {/* Status Pill */}
        <div className="inline-flex items-center gap-2.5 rounded-full border border-brand-bright/30 bg-brand-bright/10 px-4 py-2 text-xs font-semibold uppercase tracking-widest text-brand-bright animate-fade-in">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-bright opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-bright"></span>
          </span>
          <span>PRODUCTION MODEL VALIDATED • RESNET-50 STAGE B (72.88% ROC-AUC)</span>
        </div>

        {/* Title */}
        <div className="space-y-4 animate-fade-in" style={{ animationDelay: '100ms' }}>
          <h1 className="font-orbitron text-5xl sm:text-6xl lg:text-7xl font-extrabold leading-tight tracking-tight">
            Next-Gen AI <span className="gradient-text">Deepfake</span> Detection
          </h1>
          <p className="text-xl sm:text-2xl text-brand-subtle max-w-2xl mx-auto font-light leading-relaxed">
            Multi-Domain Spatial, Spectral & Temporal Forensic Analysis for Media Authenticity Verification.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 pt-4 animate-fade-in" style={{ animationDelay: '200ms' }}>
          <Link to="/detect">
            <Button leftIcon={<Cpu className="h-5 w-5" />} className="glow-effect px-8 py-4 text-base font-bold rounded-2xl">
              Start Deepfake Scan
            </Button>
          </Link>
          <Link to="/analysis/latest">
            <Button variant="secondary" leftIcon={<FileText className="h-5 w-5" />} className="px-8 py-4 text-base font-bold rounded-2xl border-brand-border">
              View Research Benchmark
            </Button>
          </Link>
        </div>
      </section>

      {/* Live System Metrics Bar */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-5xl mx-auto">
        <div className="glass-effect rounded-2xl p-6 text-center border border-brand-border/60 space-y-1 hover:border-brand-bright/40 transition-colors">
          <p className="text-3xl font-orbitron font-extrabold text-brand-bright">72.88%</p>
          <p className="text-xs uppercase font-bold tracking-wider text-brand-subtle">Validation ROC-AUC</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center border border-brand-border/60 space-y-1 hover:border-brand-bright/40 transition-colors">
          <p className="text-3xl font-orbitron font-extrabold text-emerald-400">320 Set</p>
          <p className="text-xs uppercase font-bold tracking-wider text-brand-subtle">FF++ Audited Videos</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center border border-brand-border/60 space-y-1 hover:border-brand-bright/40 transition-colors">
          <p className="text-3xl font-orbitron font-extrabold text-cyan-400">&lt; 3.5s</p>
          <p className="text-xs uppercase font-bold tracking-wider text-brand-subtle">GPU Inference Time</p>
        </div>
        <div className="glass-effect rounded-2xl p-6 text-center border border-brand-border/60 space-y-1 hover:border-brand-bright/40 transition-colors">
          <p className="text-3xl font-orbitron font-extrabold text-purple-400">100%</p>
          <p className="text-xs uppercase font-bold tracking-wider text-brand-subtle">2D FFT & Alignment</p>
        </div>
      </section>

      {/* Feature Breakdown Grid */}
      <section className="space-y-10 max-w-6xl mx-auto">
        <div className="text-center space-y-3">
          <h2 className="font-orbitron text-3xl font-bold gradient-text">Multimodal Forensic Pillars</h2>
          <p className="text-brand-subtle max-w-xl mx-auto">
            Combines deep spatial neural representations with frequency spectral transform audits to expose manipulation artifacts.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="glass-effect rounded-3xl p-8 space-y-4 border border-brand-border/60 hover:border-brand-bright/50 transition-all hover:-translate-y-1">
            <div className="p-3.5 rounded-2xl bg-brand-bright/15 text-brand-bright w-fit border border-brand-bright/20">
              <Layers className="h-7 w-7" />
            </div>
            <h3 className="font-orbitron text-xl font-bold text-brand-text">Spatial Transfer Learning</h3>
            <p className="text-sm text-brand-subtle leading-relaxed">
              Leverages ImageNet-pretrained ResNet-50 backbones with fine-tuned deeper layers (Stage B) to isolate face swap boundary blending anomalies.
            </p>
          </div>

          <div className="glass-effect rounded-3xl p-8 space-y-4 border border-brand-border/60 hover:border-brand-bright/50 transition-all hover:-translate-y-1">
            <div className="p-3.5 rounded-2xl bg-cyan-500/15 text-cyan-400 w-fit border border-cyan-500/20">
              <Activity className="h-7 w-7" />
            </div>
            <h3 className="font-orbitron text-xl font-bold text-brand-text">2D FFT Spectral Audit</h3>
            <p className="text-sm text-brand-subtle leading-relaxed">
              Transforms spatial facial crops into 2D Fast Fourier Transform log-magnitude spectra to uncover neural upsampling checkerboard grid noise.
            </p>
          </div>

          <div className="glass-effect rounded-3xl p-8 space-y-4 border border-brand-border/60 hover:border-brand-bright/50 transition-all hover:-translate-y-1">
            <div className="p-3.5 rounded-2xl bg-purple-500/15 text-purple-400 w-fit border border-purple-500/20">
              <Lock className="h-7 w-7" />
            </div>
            <h3 className="font-orbitron text-xl font-bold text-brand-text">Forensic Report Generator</h3>
            <p className="text-sm text-brand-subtle leading-relaxed">
              Generates executive, print-ready PDF forensic audit reports with anomaly localization tags, risk scores, and digital verification signatures.
            </p>
          </div>
        </div>
      </section>
    </div>
  </div>
)
