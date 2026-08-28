import { Search, Sparkles, ExternalLink, Download } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { formatDate } from '../data/mockData'
import { useDetection } from '../hooks/useDetection'
import { Badge } from '../components/ui/Badge'
import { downloadForensicReport } from '../utils/reportGenerator'

export const History = () => {
  const { detections } = useDetection()
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<'all' | 'video' | 'audio'>('all')

  const filtered = useMemo(
    () =>
      detections.filter((item) => {
        const matchedQuery = item.fileName.toLowerCase().includes(query.toLowerCase())
        const matchedType = filter === 'all' ? true : item.fileType === filter
        return matchedQuery && matchedType
      }),
    [detections, filter, query],
  )

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <header className="space-y-2">
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-4 w-4" />
          <span>Audit Logs & History</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Forensic History</h1>
        <p className="text-brand-subtle text-lg">Search, inspect, and export official forensic audit reports for past detections.</p>
      </header>

      {/* Search & Filter */}
      <div className="glass-effect rounded-2xl p-6 space-y-4 border border-brand-border/60">
        <div className="grid gap-3 md:grid-cols-[1fr_200px]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-brand-muted" />
            <input
              className="w-full rounded-xl border border-brand-border bg-brand-card/60 py-3 pl-12 pr-4 text-base text-brand-text placeholder:text-brand-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all font-medium"
              placeholder="Search by filename or ID..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>

          <select
            className="rounded-xl border border-brand-border bg-brand-card/60 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all font-medium"
            value={filter}
            onChange={(event) => setFilter(event.target.value as 'all' | 'video' | 'audio')}
            aria-label="Filter history by detection type"
          >
            <option value="all">All Types</option>
            <option value="video">Video Clips</option>
            <option value="audio">Audio Streams</option>
          </select>
        </div>
        <div className="flex items-center justify-between text-xs text-brand-muted">
          <span>Found {filtered.length} audit record{filtered.length !== 1 ? 's' : ''}</span>
          <span>Engine: ResNet-50 Fine-Tuned Stage B</span>
        </div>
      </div>

      {/* Results */}
      {filtered.length === 0 ? (
        <div className="glass-effect rounded-2xl p-12 text-center border border-brand-border/50">
          <div className="space-y-2">
            <p className="text-brand-subtle text-lg font-medium">
              {detections.length === 0
                ? 'No detection history available'
                : 'No audit records match your search query'}
            </p>
            <p className="text-brand-muted text-sm">
              {detections.length === 0
                ? 'Run a deepfake detection to populate history.'
                : 'Try clearing your search query or filters.'}
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((item) => {
            const isFake = item.verdict === 'DEEPFAKE' || item.result.includes('FAKE') || item.confidence > 50

            return (
              <article
                key={item.id}
                className="glass-effect-sm rounded-2xl border border-brand-border/60 p-5 transition hover:border-brand-bright/50 hover:bg-brand-card2/40 group"
              >
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-[2fr_1.2fr_1fr_1.2fr_auto] lg:items-center">
                  <div className="min-w-0">
                    <p className="font-semibold text-brand-text truncate text-base">{item.fileName}</p>
                    <p className="text-xs text-brand-muted mt-1 uppercase tracking-wider font-mono">
                      {item.fileType.toUpperCase()} • ID: {item.id.slice(0, 8)}
                    </p>
                  </div>

                  <div>
                    <Badge
                      label={isFake ? '🔴 DEEPFAKE (FAKE)' : '🟢 GENUINE (REAL)'}
                      tone={isFake ? 'error' : 'success'}
                    />
                  </div>

                  <div>
                    <p className="text-xs text-brand-muted uppercase tracking-wider font-semibold">Confidence</p>
                    <p className="text-lg font-bold font-orbitron text-brand-bright">{item.confidence}%</p>
                  </div>

                  <div>
                    <p className="text-xs text-brand-muted uppercase tracking-wider font-semibold">Timestamp</p>
                    <p className="text-sm text-brand-subtle">{formatDate(item.createdAt)}</p>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      variant="secondary"
                      onClick={() => downloadForensicReport(item)}
                      leftIcon={<Download className="h-4 w-4" />}
                      className="text-xs py-2 px-3 border-brand-border"
                    >
                      Report
                    </Button>
                    <Link to={`/analysis/${item.id}`}>
                      <Button variant="secondary" leftIcon={<ExternalLink className="h-4 w-4" />} className="text-xs py-2 px-3">
                        Inspect
                      </Button>
                    </Link>
                  </div>
                </div>
              </article>
            )
          })}
        </div>
      )}
    </div>
  )
}
