import { ArrowRight, Cpu, Sparkles, Play, RefreshCw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { FileUploader } from '../components/ui/FileUploader'
import { Tabs } from '../components/ui/Tabs'
import type { DetectionType } from '../types'
import { Button } from '../components/ui/Button'
import { Progress } from '../components/ui/Progress'
import { useDetection } from '../hooks/useDetection'
import { useToast } from '../hooks/useToast'
import { Badge } from '../components/ui/Badge'

export const Detect = () => {
  const [tab, setTab] = useState<DetectionType>('video')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const { status, progress, runDetection, currentResult, resetStatus } = useDetection()
  const { showToast } = useToast()

  useEffect(() => {
    if (status === 'completed') {
      showToast({ title: 'Forensic Detection complete', variant: 'success' })
    }
  }, [showToast, status])

  const handleQuickSampleSelect = (sampleName: string) => {
    const dummyFile = new File(['dummy_content'], sampleName, { type: 'video/mp4' })
    setSelectedFile(dummyFile)
    resetStatus()
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright font-semibold">
          <Sparkles className="h-4 w-4" />
          <span>Forensic Inspection Engine</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Deepfake Analyzer</h1>
        <p className="text-brand-subtle text-lg">
          Upload media or select sample clips to execute spatial & frequency forensic deepfake detection.
        </p>
      </header>

      {/* Main Upload & Scan Panel */}
      <div className="glass-effect rounded-3xl p-8 space-y-8 border border-brand-border/70 shadow-2xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-brand-border/50 pb-6">
          <Tabs
            value={tab}
            onChange={(value) => {
              setTab(value)
              setSelectedFile(null)
              resetStatus()
            }}
            items={[
              { label: 'Video Analysis', value: 'video' },
              { label: 'Audio Stream Audit', value: 'audio' },
            ]}
          />

          {/* Quick Demo Sample Selector */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-brand-muted font-semibold uppercase tracking-wider">Quick Demo:</span>
            <button
              onClick={() => handleQuickSampleSelect('FFPP_Manipulated_Sample_001.mp4')}
              className="px-3 py-1.5 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-semibold hover:bg-red-500/25 transition-colors"
            >
              Test FAKE Sample
            </button>
            <button
              onClick={() => handleQuickSampleSelect('Genuine_Interview_Sample_018.mp4')}
              className="px-3 py-1.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-semibold hover:bg-emerald-500/25 transition-colors"
            >
              Test REAL Sample
            </button>
          </div>
        </div>

        {/* Upload Zone */}
        <FileUploader acceptType={tab} file={selectedFile} onFileSelect={setSelectedFile} />

        {/* Real-time Multi-Stage Progress Scanner */}
        {(status === 'uploading' || status === 'processing' || status === 'completed') && (
          <div className="space-y-4 rounded-2xl glass-effect-sm p-6 border border-brand-border/80">
            <div className="flex items-center justify-between text-sm font-semibold">
              <div className="flex items-center gap-3">
                <RefreshCw className={`h-5 w-5 text-brand-bright ${status !== 'completed' ? 'animate-spin' : ''}`} />
                <span className="text-brand-text font-orbitron">
                  {status === 'uploading' && 'Stage 1/4: Ingesting & Extracting Frames...'}
                  {status === 'processing' && progress < 45 && 'Stage 2/4: Face Alignment & 2D FFT Transform...'}
                  {status === 'processing' && progress >= 45 && 'Stage 3/4: ResNet-50 Feature Extraction...'}
                  {status === 'completed' && 'Stage 4/4: Classification & Anomaly Localization Complete!'}
                </span>
              </div>
              <span className="text-brand-bright font-orbitron text-lg font-bold">{progress}%</span>
            </div>
            <Progress value={progress} />

            {/* Scan Pipeline Steps */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
              <div className={`p-2.5 rounded-xl border text-xs ${progress >= 25 ? 'bg-brand-bright/15 border-brand-bright/40 text-brand-bright' : 'bg-brand-card/40 border-brand-border text-brand-muted'}`}>
                ✓ Frame Extraction
              </div>
              <div className={`p-2.5 rounded-xl border text-xs ${progress >= 50 ? 'bg-brand-bright/15 border-brand-bright/40 text-brand-bright' : 'bg-brand-card/40 border-brand-border text-brand-muted'}`}>
                ✓ 2D FFT Frequency Audit
              </div>
              <div className={`p-2.5 rounded-xl border text-xs ${progress >= 75 ? 'bg-brand-bright/15 border-brand-bright/40 text-brand-bright' : 'bg-brand-card/40 border-brand-border text-brand-muted'}`}>
                ✓ ResNet-50 Deep Pass
              </div>
              <div className={`p-2.5 rounded-xl border text-xs ${progress >= 100 ? 'bg-brand-bright/15 border-brand-bright/40 text-brand-bright' : 'bg-brand-card/40 border-brand-border text-brand-muted'}`}>
                ✓ Anomaly Report
              </div>
            </div>
          </div>
        )}

        {/* Action Button */}
        <Button
          type="button"
          disabled={!selectedFile || status === 'uploading' || status === 'processing'}
          onClick={async () => {
            if (!selectedFile) return
            await runDetection(selectedFile, tab)
          }}
          className="w-full py-4 font-bold text-lg glow-effect hover:scale-[1.01] transition-transform rounded-2xl"
        >
          {status === 'uploading' || status === 'processing' ? 'Executing Forensic Scan...' : 'Start Deepfake Detection'}
          {status !== 'uploading' && status !== 'processing' && <Play className="h-5 w-5 ml-2 fill-current" />}
        </Button>
      </div>

      {/* Immediate Results Card */}
      {currentResult ? (
        <div className="glass-effect rounded-3xl p-8 space-y-6 border border-brand-border-glow shadow-2xl animate-fade-in">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <p className="text-brand-bright text-xs uppercase tracking-widest font-bold">
                Classification Verdict
              </p>
              <h2 className="text-2xl md:text-3xl font-bold text-brand-text mt-1 font-orbitron">
                {currentResult.verdict === 'DEEPFAKE' || currentResult.result.includes('FAKE')
                  ? '🔴 DEEPFAKE DETECTED (FAKE)'
                  : '🟢 MEDIA VERIFIED (GENUINE REAL)'}
              </h2>
            </div>
            <Badge
              label={
                currentResult.verdict === 'DEEPFAKE' || currentResult.result.includes('FAKE')
                  ? '🔴 DEEPFAKE'
                  : '🟢 GENUINE'
              }
              tone={
                currentResult.verdict === 'DEEPFAKE' || currentResult.result.includes('FAKE')
                  ? 'error'
                  : 'success'
              }
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="glass-effect-sm rounded-2xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors space-y-1">
              <p className="text-xs uppercase tracking-widest text-brand-muted font-bold">Verdict</p>
              <p className="text-xl font-bold font-orbitron gradient-text">{currentResult.result}</p>
            </div>
            <div className="glass-effect-sm rounded-2xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors space-y-1">
              <p className="text-xs uppercase tracking-widest text-brand-muted font-bold">Confidence Score</p>
              <p className="text-xl font-bold text-brand-bright font-orbitron">{currentResult.confidence}%</p>
            </div>
            <div className="glass-effect-sm rounded-2xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors space-y-1">
              <p className="text-xs uppercase tracking-widest text-brand-muted font-bold">Primary Anomaly Domain</p>
              <p className="text-sm font-semibold text-brand-subtle truncate">
                {currentResult.reasons && currentResult.reasons[0]
                  ? currentResult.reasons[0].location
                  : 'Spatial & Spectral Domain'}
              </p>
            </div>
          </div>

          <Link to={`/analysis/${currentResult.id}`} className="block">
            <Button variant="secondary" leftIcon={<Cpu className="h-4 w-4" />} className="w-full py-3.5 font-semibold glow-effect rounded-xl">
              Inspect Detailed Forensic Reasons & Download Report
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </Link>
        </div>
      ) : null}
    </div>
  )
}
