import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AnalysisCard } from '../components/analysis/AnalysisCard'
import { Button } from '../components/ui/Button'
import { useDetection } from '../hooks/useDetection'
import { Sparkles, ArrowLeft, Download, ShieldCheck, History } from 'lucide-react'
import { downloadForensicReport } from '../utils/reportGenerator'

export const Analysis = () => {
  const { id = '' } = useParams()
  const { detections, getDetectionById } = useDetection()

  const analysis = useMemo(() => {
    if (id === 'latest' && detections.length > 0) {
      return detections[0]
    }
    return getDetectionById(id)
  }, [getDetectionById, id, detections])

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header Bar */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brand-border/40 pb-6">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <Link to="/history">
              <Button variant="ghost" leftIcon={<ArrowLeft className="h-4 w-4" />} className="text-xs py-1.5 px-3">
                Back to History
              </Button>
            </Link>
            <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-3.5 py-1.5 text-xs font-bold uppercase tracking-widest text-brand-bright">
              <Sparkles className="h-3.5 w-3.5" />
              <span>Forensic Audit Document</span>
            </div>
          </div>

          <h1 className="text-3xl sm:text-4xl font-orbitron font-bold gradient-text">Detailed Forensic Report</h1>
          <p className="text-brand-subtle text-base max-w-2xl leading-relaxed">
            Multi-domain spatial-frequency metrics, anomaly localization, and experimental ablation matrix.
          </p>
        </div>

        {analysis ? (
          <div className="flex items-center gap-3 shrink-0 self-start md:self-auto">
            <Button
              onClick={() => downloadForensicReport(analysis)}
              leftIcon={<Download className="h-4.5 w-4.5" />}
              className="glow-effect py-3 px-6 font-bold rounded-xl text-sm"
            >
              Download PDF Report
            </Button>
          </div>
        ) : null}
      </header>

      {/* Empty State */}
      {detections.length === 0 ? (
        <div className="glass-effect rounded-3xl p-12 text-center border border-brand-border/60 shadow-xl space-y-4">
          <div className="p-4 rounded-2xl bg-brand-card/50 text-brand-bright w-fit mx-auto border border-brand-border">
            <ShieldCheck className="h-10 w-10" />
          </div>
          <div className="space-y-2 max-w-md mx-auto">
            <h3 className="text-xl font-orbitron font-bold text-brand-text">No Forensic Records Available</h3>
            <p className="text-brand-subtle text-sm">Run a deepfake detection scan first to generate a detailed report.</p>
          </div>
          <Link to="/detect" className="inline-block pt-2">
            <Button className="glow-effect px-6 py-3 font-semibold">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Launch Deepfake Analyzer
            </Button>
          </Link>
        </div>
      ) : null}

      {/* Record Not Found */}
      {detections.length > 0 && !analysis ? (
        <div className="glass-effect rounded-3xl p-12 text-center border border-brand-border/60 shadow-xl space-y-4">
          <div className="p-4 rounded-2xl bg-brand-card/50 text-amber-400 w-fit mx-auto border border-brand-border">
            <History className="h-10 w-10" />
          </div>
          <div className="space-y-2 max-w-md mx-auto">
            <h3 className="text-xl font-orbitron font-bold text-brand-text">Analysis Record Not Found</h3>
            <p className="text-brand-subtle text-sm">The requested report ID could not be located in current session memory.</p>
          </div>
          <Link to={`/analysis/${detections[0].id}`} className="inline-block pt-2">
            <Button variant="secondary" className="px-6 py-3 font-semibold border-brand-border">
              View Latest Available Audit Report
            </Button>
          </Link>
        </div>
      ) : null}

      {/* Analysis Report View */}
      {analysis ? <AnalysisCard analysis={analysis} /> : null}
    </div>
  )
}
