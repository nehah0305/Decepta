import { ArrowRight, Cpu, Sparkles } from 'lucide-react'
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
      showToast({ title: 'Detection complete', variant: 'success' })
    }
  }, [showToast, status])

  return (
    <div className="space-y-8 animate-fade-in">
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-4 w-4" />
          <span>Detection Engine</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Deepfake Detection</h1>
        <p className="text-brand-subtle text-lg">Upload media and launch a simulated detection run with our advanced AI.</p>
      </header>

      {/* Upload Card */}
      <div className="glass-effect rounded-2xl p-8 space-y-6">
        <Tabs
          value={tab}
          onChange={(value) => {
            setTab(value)
            setSelectedFile(null)
            resetStatus()
          }}
          items={[
            { label: 'Video', value: 'video' },
            { label: 'Audio', value: 'audio' },
          ]}
        />

        <FileUploader acceptType={tab} file={selectedFile} onFileSelect={setSelectedFile} />

        {(status === 'uploading' || status === 'processing' || status === 'completed') && (
          <div className="space-y-3 rounded-xl glass-effect-sm p-4 border border-brand-border">
            <div className="flex items-center justify-between text-sm">
              <span className="text-brand-text font-medium">
                {status === 'uploading' && 'Uploading file...'}
                {status === 'processing' && 'Processing detection...'}
                {status === 'completed' && 'Detection complete'}
              </span>
              <span className="text-brand-bright font-semibold">{progress}%</span>
            </div>
            <Progress value={progress} />
          </div>
        )}

        <Button
          type="button"
          disabled={!selectedFile || status === 'uploading' || status === 'processing'}
          onClick={async () => {
            if (!selectedFile) return
            await runDetection(selectedFile, tab)
          }}
          className="w-full py-3 font-semibold text-base glow-effect hover:scale-105 transition-transform"
        >
          {status === 'uploading' || status === 'processing' ? 'Processing...' : 'Start Detection'}
          {status !== 'uploading' && status !== 'processing' && <ArrowRight className="h-5 w-5" />}
        </Button>
      </div>

      {/* Results Card */}
      {currentResult ? (
        <div className="glass-effect rounded-2xl p-8 space-y-6 border border-brand-bright/20 animate-fade-in">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-brand-subtle text-sm uppercase tracking-widest">Detection Result</p>
              <h2 className="text-2xl font-semibold text-brand-text mt-1">Analysis Complete</h2>
            </div>
            <Badge label={currentResult.status} tone="success" />
          </div>
          
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="glass-effect-sm rounded-xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors">
              <p className="text-xs uppercase tracking-widest text-brand-muted">Status</p>
              <p className="mt-3 text-2xl font-semibold gradient-text">{currentResult.result}</p>
            </div>
            <div className="glass-effect-sm rounded-xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors">
              <p className="text-xs uppercase tracking-widest text-brand-muted">Confidence Score</p>
              <p className="mt-3 text-2xl font-semibold text-brand-bright">{currentResult.confidence}%</p>
            </div>
            <div className="glass-effect-sm rounded-xl p-5 border border-brand-border hover:border-brand-bright/50 transition-colors">
              <p className="text-xs uppercase tracking-widest text-brand-muted">Processing Time</p>
              <p className="mt-3 text-2xl font-semibold text-brand-bright">{currentResult.processingTime}s</p>
            </div>
          </div>
          
          <Link to={`/analysis/${currentResult.id}`} className="block">
            <Button variant="secondary" leftIcon={<Cpu className="h-4 w-4" />} className="w-full py-3 font-semibold glow-effect">
              View Detailed Analysis
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      ) : null}
    </div>
  )
}
