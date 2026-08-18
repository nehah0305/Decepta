import { ArrowRight, Sparkles, Zap, TrendingUp, Eye } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'

export const Landing = () => (
  <div className="relative min-h-screen overflow-hidden bg-brand-bg text-brand-text">
    {/* Animated Background Gradients */}
    <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(80,214,209,0.25),transparent_38%),radial-gradient(circle_at_90%_10%,rgba(38,184,181,0.28),transparent_36%),linear-gradient(135deg,rgba(24,62,68,0.3),transparent)]"></div>
    
    {/* Floating Orbs */}
    <div className="pointer-events-none absolute -top-40 -left-40 h-96 w-96 rounded-full bg-brand-bright/10 blur-3xl"></div>
    <div className="pointer-events-none absolute -bottom-32 -right-32 h-80 w-80 rounded-full bg-brand-primary/10 blur-3xl"></div>
    
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-30"></div>
    
    <div className="relative mx-auto flex w-full max-w-6xl flex-col px-6 py-16 lg:py-28">
      {/* Hero Section */}
      <div className="animate-fade-in max-w-3xl">
        {/* Badge */}
        <div className="mb-6 inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright">
          <Sparkles className="h-4 w-4" />
          <span>AI Detection Suite</span>
        </div>
        
        {/* Title with Unique Font and Gradient */}
        <div className="space-y-4">
          <h1 className="font-orbitron text-5xl leading-tight sm:text-6xl lg:text-7xl">
            <span className="gradient-text">Decepta</span>
          </h1>
          <p className="text-2xl font-light text-brand-subtle sm:text-3xl">
            Deepfake Detection Reimagined
          </p>
        </div>
        
        {/* Description */}
        <p className="mt-6 max-w-2xl text-base text-brand-muted sm:text-lg leading-relaxed">
          Advanced AI-powered platform for detecting and analyzing deepfake content with precision. 
          Upload, analyze, and gain insights into media authenticity using cutting-edge machine learning technology.
        </p>
        
        {/* CTA Buttons */}
        <div className="mt-10 flex flex-wrap gap-4">
          <Link to="/login">
            <Button 
              leftIcon={<ArrowRight className="h-4 w-4" />}
              className="glow-effect"
            >
              Start Detecting
            </Button>
          </Link>
          <a href="#features">
            <Button variant="secondary">Explore Features</Button>
          </a>
        </div>
      </div>

      {/* Features Section */}
      <div id="features" className="mt-24 space-y-4">
        <h2 className="text-3xl font-semibold text-brand-text mb-8">Powerful Features</h2>
        <div className="grid gap-6 md:grid-cols-3">
          {[
            { 
              icon: Zap,
              title: 'Lightning Fast Detection', 
              text: 'Analyze media files in seconds with our optimized detection engine.' ,
              delayClass: ''
            },
            { 
              icon: TrendingUp,
              title: 'Detailed Analytics', 
              text: 'View comprehensive metrics, timelines, and confidence scores for each analysis.' ,
              delayClass: 'md:mt-6'
            },
            { 
              icon: Eye,
              title: 'Smart Insights', 
              text: 'Track detection history and identify patterns across multiple analyses.' ,
              delayClass: ''
            },
          ].map((item) => {
            const Icon = item.icon
            return (
              <div 
                key={item.title} 
                className={`group animate-fade-in glass-effect rounded-2xl p-6 transition-all duration-300 hover:glass-effect-lg hover:shadow-lg cursor-pointer ${item.delayClass}`}
              >
                {/* Icon Container */}
                <div className="mb-4 inline-block rounded-xl bg-brand-bright/10 p-3 transition-colors duration-300 group-hover:bg-brand-bright/20">
                  <Icon className="h-6 w-6 text-brand-bright" />
                </div>
                
                {/* Content */}
                <h3 className="text-lg font-semibold text-brand-text mb-2 group-hover:text-brand-bright transition-colors duration-300">
                  {item.title}
                </h3>
                <p className="text-sm text-brand-subtle leading-relaxed">
                  {item.text}
                </p>
                
                {/* Hover Arrow */}
                <div className="mt-4 inline-flex items-center gap-2 text-brand-bright opacity-0 transition-all duration-300 group-hover:opacity-100">
                  <span className="text-xs font-semibold">Learn More</span>
                  <ArrowRight className="h-3 w-3 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Stats Section */}
      <div className="mt-24 glass-effect rounded-2xl p-8 md:p-12">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {[
            { label: 'Analyses', value: '10K+' },
            { label: 'Accuracy', value: '99%' },
            { label: 'Users', value: '5K+' },
            { label: 'Uptime', value: '99.9%' },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <p className="text-3xl font-orbitron text-brand-bright mb-2">
                {stat.value}
              </p>
              <p className="text-sm text-brand-subtle">{stat.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Footer CTA */}
      <div className="mt-24 text-center space-y-6">
        <p className="text-brand-subtle">Ready to detect deepfakes with confidence?</p>
        <Link to="/login">
          <Button 
            className="glow-effect font-semibold px-8 py-3"
          >
            Get Started Now <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </Link>
      </div>
    </div>
  </div>
)
