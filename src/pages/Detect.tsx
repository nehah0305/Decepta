import { ArrowRight, Cpu } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Card } from '../components/ui/Card'
import { FileUploader } from '../components/ui/FileUploader'
import { Tabs } from '../components/ui/Tabs'
import { DetectionType } from '../types'
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
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-3xl font-semibold text-brand-text">Detect</h1>
        <p className="mt-1 text-sm text-brand-subtle">Upload media and launch a simulated detection run.</p>
      </header>

      <Card className="space-y-5">
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
          <div className="space-y-2 rounded-xl border border-brand-border bg-brand-card2/45 p-3">
            <div className="flex items-center justify-between text-sm text-brand-subtle">
              <span>
                {status === 'uploading' && 'Uploading file...'}
                {status === 'processing' && 'Processing detection...'}
                {status === 'completed' && 'Detection complete'}
              </span>
              <span>{progress}%</span>
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
        >
          {status === 'uploading' || status === 'processing' ? 'Processing...' : 'Detect'}
        </Button>
      </Card>

      {currentResult ? (
        <Card className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <p className="text-lg font-semibold text-brand-text">Detection Result</p>
            <Badge label={currentResult.status} tone="success" />
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-brand-border bg-brand-card2/40 p-3">
              <p className="text-xs text-brand-muted">Status</p>
              <p className="mt-1 text-base font-semibold text-brand-text">{currentResult.result}</p>
            </div>
            <div className="rounded-xl border border-brand-border bg-brand-card2/40 p-3">
              <p className="text-xs text-brand-muted">Confidence</p>
              <p className="mt-1 text-base font-semibold text-brand-text">{currentResult.confidence}%</p>
            </div>
            <div className="rounded-xl border border-brand-border bg-brand-card2/40 p-3">
              <p className="text-xs text-brand-muted">Processing Time</p>
              <p className="mt-1 text-base font-semibold text-brand-text">{currentResult.processingTime}s</p>
            </div>
          </div>
          <Link to={`/analysis/${currentResult.id}`}>
            <Button variant="secondary" leftIcon={<Cpu className="h-4 w-4" />}>
              View analysis
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </Card>
      ) : null}
    </div>
  )
}
