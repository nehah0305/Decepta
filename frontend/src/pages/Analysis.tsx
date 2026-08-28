import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AnalysisCard } from '../components/analysis/AnalysisCard'
import { Button } from '../components/ui/Button'
import { useDetection } from '../hooks/useDetection'
import { Sparkles, ArrowLeft, Download } from 'lucide-react'
import { downloadForensicReport } from '../utils/reportGenerator'

export const Analysis = () => {
  const { id = '' } = useParams()
  const { detections, getDetectionById } = useDetection()

  const analysis = useMemo(() => getDetectionById(id), [getDetectionById, id])

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
            <Sparkles className="h-4 w-4" />
            <span>Detailed Forensic Analysis</span>
          </div>
          <h1 className="text-4xl font-orbitron font-bold gradient-text">Detection Report</h1>
          <p className="text-brand-subtle text-lg">Comprehensive spatial-frequency confidence metrics and anomaly localization.</p>
        </div>

        {analysis ? (
          <Button
            onClick={() => downloadForensicReport(analysis)}
            leftIcon={<Download className="h-4 w-4" />}
            className="glow-effect py-3 px-5 self-start sm:self-auto font-semibold shrink-0"
          >
            Download Forensic Report
          </Button>
        ) : null}
      </header>

      {detections.length === 0 ? (
        <div className="glass-effect rounded-2xl p-12 text-center border border-brand-border/50">
          <div className="space-y-4">
            <p className="text-brand-subtle text-lg font-medium">No analysis available yet</p>
            <p className="text-brand-muted">Run a detection first to see detailed analysis results.</p>
            <Link to="/detect" className="inline-block">
              <Button className="glow-effect">
                <ArrowLeft className="h-4 w-4" />
                Start Detection
              </Button>
            </Link>
          </div>
        </div>
      ) : null}

      {detections.length > 0 && !analysis ? (
        <div className="glass-effect rounded-2xl p-12 text-center border border-brand-border/50">
          <div className="space-y-4">
            <p className="text-brand-subtle text-lg font-medium">Analysis record not found</p>
            <p className="text-brand-muted">The analysis you're looking for doesn't exist.</p>
            <Link to={`/analysis/${detections[0].id}`} className="inline-block">
              <Button variant="secondary">View Latest Analysis</Button>
            </Link>
          </div>
        </div>
      ) : null}

      {analysis ? <AnalysisCard analysis={analysis} /> : null}
    </div>
  )
}
