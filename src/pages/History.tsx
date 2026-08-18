import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
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
    <div className="space-y-6 animate-fade-in">
      <header>
        <h1 className="text-3xl font-semibold text-brand-text">History</h1>
        <p className="mt-1 text-sm text-brand-subtle">Search and inspect previously processed detections.</p>
      </header>

      <Card className="space-y-4">
        <div className="grid gap-3 md:grid-cols-[1fr_180px]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-brand-muted" />
            <input
              className="w-full rounded-xl border border-brand-border bg-brand-card2/55 py-2.5 pl-9 pr-3 text-sm text-brand-text placeholder:text-brand-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright"
              placeholder="Search history..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>

          <select
            className="rounded-xl border border-brand-border bg-brand-card2/55 px-3 text-sm text-brand-subtle focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright"
            value={filter}
            onChange={(event) => setFilter(event.target.value as 'all' | 'video' | 'audio')}
            aria-label="Filter history by detection type"
          >
            <option value="all">All Types</option>
            <option value="video">Video</option>
            <option value="audio">Audio</option>
          </select>
        </div>

        {filtered.length === 0 ? (
          <div className="rounded-xl border border-brand-border bg-brand-card2/35 p-8 text-center">
            <p className="text-brand-subtle">
              {detections.length === 0
                ? 'No detection history yet. Run a detection to get started.'
                : 'No results match your search.'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filtered.map((item) => (
              <article
                key={item.id}
                className="rounded-xl border border-brand-border bg-brand-card2/40 p-4 transition hover:border-brand-bright/35"
              >
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-[2fr_repeat(4,1fr)] lg:items-center">
                  <div>
                    <p className="font-medium text-brand-text">{item.fileName}</p>
                    <p className="text-xs text-brand-muted">{item.fileType.toUpperCase()}</p>
                  </div>
                  <p className="text-sm text-brand-subtle">{formatDate(item.createdAt)}</p>
                  <p className="text-sm text-brand-subtle">{item.status}</p>
                  <p className="text-sm font-semibold text-brand-bright">{item.confidence}%</p>
                  <div>
                    <Link to={`/analysis/${item.id}`}>
                      <Button variant="secondary" className="w-full sm:w-auto">
                        View Analysis
                      </Button>
                    </Link>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}
