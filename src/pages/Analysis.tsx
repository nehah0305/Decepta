import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AnalysisCard } from '../components/analysis/AnalysisCard'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { useDetection } from '../hooks/useDetection'

export const Analysis = () => {
  const { id = '' } = useParams()
  const { detections, getDetectionById } = useDetection()

  const analysis = useMemo(() => getDetectionById(id), [getDetectionById, id])

  return (
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-3xl font-semibold text-brand-text">Analysis</h1>
        <p className="mt-1 text-sm text-brand-subtle">Detailed confidence metrics and detection timeline.</p>
      </header>

      {detections.length === 0 ? (
        <Card className="text-center">
          <p className="text-brand-subtle">No analysis available yet. Run a detection first.</p>
          <Link to="/detect" className="mt-3 inline-block">
            <Button>Go to Detect</Button>
          </Link>
        </Card>
      ) : null}

      {detections.length > 0 && !analysis ? (
        <Card className="text-center">
          <p className="text-brand-subtle">Analysis record not found.</p>
          <Link to={`/analysis/${detections[0].id}`} className="mt-3 inline-block">
            <Button variant="secondary">Open Latest</Button>
          </Link>
        </Card>
      ) : null}

      {analysis ? <AnalysisCard analysis={analysis} /> : null}
    </div>
  )
}
