import { formatDate } from '../../data/mockData'
import type { DetectionRecord } from '../../types'
import { Badge } from '../ui/Badge'
import { Card } from '../ui/Card'
import { Button } from '../ui/Button'
import { Download, FileText, Cpu, Clock, CheckCircle2, RefreshCw } from 'lucide-react'
import { ConfidenceScore } from './ConfidenceScore'
import { DetectionTimeline } from './DetectionTimeline'
import { MetricsCard } from './MetricsCard'
import { AblationTable } from './AblationTable'
import { ForensicReasonsCard } from './ForensicReasonsCard'
import { downloadForensicReport } from '../../utils/reportGenerator'
import { Link } from 'react-router-dom'

interface AnalysisCardProps {
  analysis: DetectionRecord
}

export const AnalysisCard = ({ analysis }: AnalysisCardProps) => {
  const timeline = [
    { label: 'Initial Frame Decode', start: 0, end: 20, active: true },
    { label: '2D FFT Spectral Scan', start: 20, end: 58, active: true },
    { label: 'ResNet-50 Deep Pass', start: 58, end: 84, active: true },
    { label: 'Anomaly Localization', start: 84, end: 100, active: true },
  ]

  const isFake = analysis.verdict === 'DEEPFAKE' || analysis.result.includes('FAKE') || analysis.confidence > 50

  return (
    <div className="space-y-8">
      {/* Classification Verdict & Detailed Forensic Anomaly Cards */}
      <ForensicReasonsCard
        verdict={analysis.verdict}
        resultLabel={analysis.result}
        confidence={analysis.confidence}
        reasons={analysis.reasons}
      />

      {/* Main Forensic Analysis Record Card */}
      <Card className="space-y-6 border border-brand-border/70 shadow-2xl p-6 sm:p-8">
        {/* Record Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-brand-border/40 pb-5">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-brand-muted font-bold">Audit Target Record</p>
            <h2 className="mt-1 text-2xl font-orbitron font-bold text-brand-text">{analysis.fileName}</h2>
            <p className="mt-1 text-xs text-brand-subtle font-mono">
              TYPE: {analysis.fileType.toUpperCase()} • ID: {analysis.id} • DATE: {formatDate(analysis.createdAt)}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Badge
              label={isFake ? '🔴 DEEPFAKE (FAKE)' : '🟢 GENUINE (REAL)'}
              tone={isFake ? 'error' : 'success'}
            />
            <Button
              variant="secondary"
              onClick={() => downloadForensicReport(analysis)}
              leftIcon={<Download className="h-4 w-4" />}
              className="text-xs py-2 px-3.5 border-brand-border"
            >
              Export PDF Report
            </Button>
            <Link to="/detect">
              <Button variant="ghost" leftIcon={<RefreshCw className="h-4 w-4" />} className="text-xs py-2 px-3">
                Re-Scan
              </Button>
            </Link>
          </div>
        </div>

        {/* Gauge + Metrics Grid */}
        <div className="grid gap-6 grid-cols-1 lg:grid-cols-[240px_1fr] items-start">
          {/* Gauge Box */}
          <div className="flex flex-col items-center justify-center rounded-2xl border border-brand-border bg-brand-card/50 p-6 space-y-3 shadow-inner">
            <ConfidenceScore confidence={analysis.confidence} />
            <div className="text-center space-y-0.5">
              <p className="text-xs font-bold uppercase tracking-wider text-brand-bright">
                {isFake ? 'High Artificial Variance' : 'High Organic Coherence'}
              </p>
              <p className="text-[11px] text-brand-muted">ResNet-50 Stage B Audit</p>
            </div>
          </div>

          {/* Metrics & Timeline Grid */}
          <div className="space-y-6">
            <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
              <MetricsCard
                label="Classification"
                value={isFake ? 'DEEPFAKE' : 'GENUINE'}
                icon={<FileText className="h-4 w-4" />}
              />
              <MetricsCard
                label="Confidence"
                value={`${analysis.confidence}%`}
                icon={<CheckCircle2 className="h-4 w-4" />}
              />
              <MetricsCard
                label="Inference Time"
                value={`${analysis.processingTime}s`}
                icon={<Clock className="h-4 w-4" />}
              />
              <MetricsCard
                label="Model Engine"
                value={analysis.modelVersion}
                icon={<Cpu className="h-4 w-4" />}
              />
            </div>

            {/* Timeline Box */}
            <div className="rounded-2xl border border-brand-border bg-brand-card/40 p-5 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs uppercase tracking-wider text-brand-muted font-bold">
                  Inference Scan Timeline
                </p>
                <span className="text-[11px] text-brand-bright font-mono font-semibold">100% Complete</span>
              </div>
              <DetectionTimeline events={timeline} />
            </div>
          </div>
        </div>
      </Card>

      {/* Visual Ablation Benchmark Matrix */}
      <AblationTable />
    </div>
  )
}
