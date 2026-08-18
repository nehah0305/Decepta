import { ArrowRight, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'

export const Landing = () => (
  <div className="relative min-h-screen overflow-hidden bg-brand-bg text-brand-text">
    <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(80,214,209,0.18),transparent_38%),radial-gradient(circle_at_90%_10%,rgba(38,184,181,0.22),transparent_36%),linear-gradient(135deg,rgba(24,62,68,0.25),transparent)]"></div>
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-45"></div>
    <div className="relative mx-auto flex w-full max-w-6xl flex-col px-6 py-20 lg:py-28">
      <div className="animate-fade-in max-w-3xl">
        <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-brand-border bg-brand-card2/50 px-3 py-1 text-xs uppercase tracking-wider text-brand-subtle">
          <Sparkles className="h-3.5 w-3.5" /> AI Detection Suite
        </p>
        <h1 className="text-4xl font-semibold leading-tight sm:text-5xl lg:text-6xl">Project Name</h1>
        <p className="mt-4 max-w-2xl text-base text-brand-subtle sm:text-lg">
          Futuristic project-analysis and detection platform for high-precision audio/video intelligence workflows.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link to="/login">
            <Button leftIcon={<ArrowRight className="h-4 w-4" />}>Get Started</Button>
          </Link>
          <a href="#features">
            <Button variant="secondary">Learn More</Button>
          </a>
        </div>
      </div>

      <div id="features" className="mt-16 grid gap-4 md:grid-cols-3">
        {[
          { title: 'Upload & Detect', text: 'Drag-drop media with live processing states and confidence scoring.' },
          { title: 'Historical Insights', text: 'Search, filter, and reopen previous detections in seconds.' },
          { title: 'Technical Analysis', text: 'Visual timelines and metric cards for deeper investigation.' },
        ].map((item) => (
          <Card key={item.title} className="animate-fade-in space-y-2 p-5">
            <h2 className="text-lg font-semibold text-brand-text">{item.title}</h2>
            <p className="text-sm text-brand-subtle">{item.text}</p>
          </Card>
        ))}
      </div>
    </div>
  </div>
)
