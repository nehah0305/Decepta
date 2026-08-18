import { Search, Sparkles, ExternalLink } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { formatDate } from '../data/mockData'
import { useDetection } from '../hooks/useDetection'

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
          <span>Detection History</span>
        </div>
        <h1 className="text-4xl font-orbitron font-bold gradient-text">Analysis History</h1>
        <p className="text-brand-subtle text-lg">Search and inspect all your previously processed detections.</p>
      </header>

      {/* Search & Filter */}
      <div className="glass-effect rounded-2xl p-6 space-y-4">
        <div className="grid gap-3 md:grid-cols-[1fr_180px]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-brand-muted" />
            <input
              className="w-full rounded-xl border border-brand-border bg-brand-card2/40 py-3 pl-12 pr-4 text-base text-brand-text placeholder:text-brand-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
              placeholder="Search by filename..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>

          <select
            className="rounded-xl border border-brand-border bg-brand-card2/40 px-4 py-3 text-base text-brand-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright transition-all"
            value={filter}
            onChange={(event) => setFilter(event.target.value as 'all' | 'video' | 'audio')}
            aria-label="Filter history by detection type"
          >
            <option value="all">All Types</option>
            <option value="video">Video</option>
            <option value="audio">Audio</option>
          </select>
        </div>
        <p className="text-xs text-brand-muted">Found {filtered.length} result{filtered.length !== 1 ? 's' : ''}</p>
      </div>

      {/* Results */}
      {filtered.length === 0 ? (
        <div className="glass-effect rounded-2xl p-12 text-center border border-brand-border/50">
          <div className="space-y-2">
            <p className="text-brand-subtle text-lg font-medium">
              {detections.length === 0
                ? 'No detection history yet'
                : 'No results match your search'}
            </p>
            <p className="text-brand-muted text-sm">
              {detections.length === 0
                ? 'Run a detection to get started.'
                : 'Try adjusting your search filters.'}
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((item) => (
            <article
              key={item.id}
              className="glass-effect-sm rounded-xl border border-brand-border p-5 transition hover:border-brand-bright/50 hover:bg-brand-card2/60 group"
            >
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-[2fr_repeat(4,1fr)_auto] lg:items-center">
                <div className="min-w-0">
                  <p className="font-semibold text-brand-text truncate">{item.fileName}</p>
                  <p className="text-xs text-brand-muted mt-1 uppercase tracking-wider">{item.fileType.toUpperCase()}</p>
                </div>
                <div>
                  <p className="text-sm text-brand-subtle">{formatDate(item.createdAt)}</p>
                </div>
                <div>
                  <span className="inline-block px-3 py-1 rounded-lg bg-brand-primary/20 text-xs font-medium text-brand-bright border border-brand-primary/30">
                    {item.status}
                  </span>
                </div>
                <div>
                  <p className="text-base font-semibold gradient-text">{item.confidence}%</p>
                </div>
                <div>
                  <Link to={`/analysis/${item.id}`}>
                    <Button variant="secondary" leftIcon={<ExternalLink className="h-4 w-4" />} className="w-full sm:w-auto text-sm px-4 py-2">
                      View
                    </Button>
                  </Link>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
