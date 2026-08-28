import { formatDate } from '../../data/mockData'
import type { DetectionRecord } from '../../types'
import { Badge } from '../ui/Badge'
import { Card } from '../ui/Card'
import { ConfidenceScore } from './ConfidenceScore'
import { DetectionTimeline } from './DetectionTimeline'
import { MetricsCard } from './MetricsCard'
import { AblationTable } from './AblationTable'

interface AnalysisCardProps {
  analysis: DetectionRecord
}

export const AnalysisCard = ({ analysis }: AnalysisCardProps) => {
  const timeline = [
    { label: 'Initial Scan', start: 0, end: 20, active: false },
    { label: 'Signal Pattern', start: 20, end: 58, active: true },
    { label: 'Anomaly Burst', start: 58, end: 84, active: true },
    { label: 'Stabilization', start: 84, end: 100, active: false },
  ]

  return (
    <div className="space-y-8">
      <Card className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-brand-muted">Analysis name</p>
            <h2 className="mt-1 text-2xl font-semibold text-brand-text">{analysis.fileName}</h2>
            <p className="mt-1 text-sm text-brand-subtle">
              {analysis.fileType.toUpperCase()} • {formatDate(analysis.createdAt)}
            </p>
          </div>
          <Badge label={analysis.result} tone={analysis.result === 'Detected' ? 'success' : 'warning'} />
        </div>

        <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
          <div className="flex items-center justify-center rounded-2xl border border-brand-border bg-brand-card2/40 p-4">
            <ConfidenceScore confidence={analysis.confidence} />
          </div>

          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricsCard label="Confidence" value={`${analysis.confidence}%`} />
              <MetricsCard label="Processing Time" value={`${analysis.processingTime}s`} />
              <MetricsCard label="Frames/Segments" value={String(analysis.segments)} />
              <MetricsCard label="Model Version" value={analysis.modelVersion} />
            </div>

            <div className="rounded-xl border border-brand-border bg-brand-card2/35 p-4">
              <p className="mb-3 text-xs uppercase tracking-wide text-brand-muted">Timeline</p>
              <DetectionTimeline events={timeline} />
            </div>
          </div>
        </div>
      </Card>

      {/* Visual Ablation Benchmark Table */}
      <AblationTable />
    </div>
  )
}
