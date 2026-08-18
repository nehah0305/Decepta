import { ArrowRight, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'

export const Landing = () => (
  <div className="relative min-h-screen overflow-hidden bg-brand-bg text-brand-text">
    {/* Background Pattern */}
    <div className="absolute inset-0 opacity-40">
      <img 
        src="/Background pattern decorative.png" 
        alt="background pattern" 
        className="h-full w-full object-cover"
      />
    </div>

    {/* Decorative Lines/Shapes - Right side */}
    <div className="absolute inset-0 overflow-hidden">
      <img 
        src="/Frame 3.png" 
        alt="decorative shapes" 
        className="absolute right-0 top-0 h-full w-1/2 object-cover object-left"
      />
    </div>

    {/* Main Content */}
    <div className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-center px-6 py-20 lg:py-28 min-h-screen">
      {/* Center Content */}
      <div className="w-full lg:w-3/4 space-y-8 animate-fade-in text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright mx-auto">
          <Sparkles className="h-4 w-4" />
          <span>AI Detection Suite</span>
        </div>

        {/* Hero Container with Glass Effect */}
        <div className="space-y-6">
          {/* Title */}
          <div>
            <h1 className="font-orbitron text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight">
              <span className="gradient-text">Decepta</span>
            </h1>
            <p className="text-xl sm:text-2xl text-brand-subtle mt-2 font-light">
              Deepfake Detection Reimagined
            </p>
          </div>

          {/* Description */}
          <p className="text-base sm:text-lg text-brand-muted leading-relaxed mx-auto max-w-2xl">
            Advanced AI-powered platform for detecting and analyzing deepfake content with precision. Upload, analyze, and gain insights into media authenticity.
          </p>
        </div>

        {/* CTA Button */}
        <div className="flex justify-center">
          <Link to="/login">
            <Button 
              leftIcon={<ArrowRight className="h-4 w-4" />}
              className="glow-effect text-base px-6 py-3 font-semibold"
            >
              Get Started
            </Button>
          </Link>
        </div>
      </div>

      {/* Right Side - Visual Element */}
      <div className="hidden lg:block lg:w-1/4" />
    </div>
  </div>
)
