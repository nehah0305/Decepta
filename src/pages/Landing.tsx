import { ArrowRight, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'

export const Landing = () => (
  <div className="relative min-h-screen overflow-hidden text-brand-text" style={{
    background: 'linear-gradient(135deg, #0C111D 0%, #142633 50%, #0C111D 100%), radial-gradient(circle at 12% 16%, rgba(38, 184, 181, 0.16), transparent 40%), radial-gradient(circle at 88% 7%, rgba(80, 214, 209, 0.14), transparent 32%), radial-gradient(circle at 50% 100%, rgba(38, 184, 181, 0.08), transparent 50%)',
    backgroundAttachment: 'fixed'
  }}>
    {/* Animated Background Gradient */}
    <div className="absolute inset-0 opacity-40 animate-shimmer">
      <img 
        src="/Background pattern decorative.png" 
        alt="background pattern" 
        className="h-full w-full object-cover"
      />
    </div>

    {/* Decorative Lines/Shapes - Right side with animation */}
    <div className="absolute inset-0 overflow-hidden">
      <img 
        src="/Frame 3.png" 
        alt="decorative shapes" 
        className="absolute right-0 top-0 h-full w-1/2 object-cover object-left animate-pulse-glow"
      />
    </div>

    {/* Animated Floating Orbs */}
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Top-left floating orb */}
      <div className="absolute -top-48 -left-48 h-96 w-96 rounded-full bg-brand-bright/8 blur-3xl animate-float-slow"></div>
      
      {/* Top-right floating orb */}
      <div className="absolute -top-32 -right-32 h-80 w-80 rounded-full bg-brand-primary/10 blur-3xl animate-float-slower" style={{animationDelay: '2s'}}></div>
      
      {/* Bottom-left floating orb */}
      <div className="absolute -bottom-40 -left-32 h-72 w-72 rounded-full bg-brand-bright/6 blur-3xl animate-float" style={{animationDelay: '1s'}}></div>
      
      {/* Bottom-right floating orb */}
      <div className="absolute -bottom-24 -right-48 h-96 w-96 rounded-full bg-brand-primary/8 blur-3xl animate-glow-drift"></div>
    </div>

    {/* Animated Grid Overlay */}
    <div className="absolute inset-0 bg-grid-overlay opacity-20 animate-shimmer" style={{animationDelay: '0.5s'}}></div>

    {/* Twinkling Stars */}
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Star 1 */}
      <div className="absolute top-1/4 left-1/6 w-1 h-1 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '0s'}}></div>
      
      {/* Star 2 */}
      <div className="absolute top-1/3 right-1/5 w-0.5 h-0.5 bg-brand-bright rounded-full animate-twinkle-slow" style={{animationDelay: '0.5s'}}></div>
      
      {/* Star 3 */}
      <div className="absolute top-2/3 left-1/3 w-1 h-1 bg-brand-bright rounded-full animate-twinkle-slower" style={{animationDelay: '1s'}}></div>
      
      {/* Star 4 */}
      <div className="absolute top-1/2 right-1/3 w-0.5 h-0.5 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '1.5s'}}></div>
      
      {/* Star 5 */}
      <div className="absolute bottom-1/4 left-2/3 w-1 h-1 bg-brand-bright rounded-full animate-twinkle-slow" style={{animationDelay: '0.3s'}}></div>
      
      {/* Star 6 */}
      <div className="absolute top-1/6 right-1/3 w-0.5 h-0.5 bg-brand-primary rounded-full animate-twinkle-slower" style={{animationDelay: '2s'}}></div>
      
      {/* Star 7 */}
      <div className="absolute bottom-1/3 right-1/4 w-1 h-1 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '0.8s'}}></div>
      
      {/* Star 8 */}
      <div className="absolute top-3/4 right-1/2 w-0.5 h-0.5 bg-brand-primary rounded-full animate-shimmer-star" style={{animationDelay: '1.2s'}}></div>
      
      {/* Star 9 - Additional */}
      <div className="absolute top-1/5 left-1/3 w-1 h-1 bg-brand-bright rounded-full animate-twinkle-slow" style={{animationDelay: '1.8s'}}></div>
      
      {/* Star 10 */}
      <div className="absolute top-4/5 left-1/5 w-0.5 h-0.5 bg-brand-primary rounded-full animate-twinkle" style={{animationDelay: '0.2s'}}></div>
      
      {/* Star 11 */}
      <div className="absolute top-1/2 left-1/2 w-1 h-1 bg-brand-bright rounded-full animate-shimmer-star" style={{animationDelay: '2.5s'}}></div>
      
      {/* Star 12 */}
      <div className="absolute bottom-1/3 left-1/4 w-0.5 h-0.5 bg-brand-primary rounded-full animate-twinkle-slower" style={{animationDelay: '1.1s'}}></div>
      
      {/* Star 13 */}
      <div className="absolute top-3/4 right-1/3 w-1 h-1 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '0.6s'}}></div>
      
      {/* Star 14 */}
      <div className="absolute top-1/3 left-3/4 w-0.5 h-0.5 bg-brand-primary rounded-full animate-twinkle-slow" style={{animationDelay: '1.9s'}}></div>
      
      {/* Star 15 */}
      <div className="absolute bottom-2/3 right-2/3 w-1 h-1 bg-brand-bright rounded-full animate-shimmer-star" style={{animationDelay: '0.7s'}}></div>
      
      {/* Star 16 */}
      <div className="absolute top-2/5 right-1/4 w-0.5 h-0.5 bg-brand-primary rounded-full animate-twinkle-slower" style={{animationDelay: '2.2s'}}></div>
    </div>

    {/* Floating Particles - Unique Animation */}
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Particle 1 */}
      <div className="absolute top-1/4 left-1/4 w-1.5 h-1.5 bg-brand-bright/40 rounded-full blur-sm animate-particle-float" style={{animationDelay: '0s'}}></div>
      
      {/* Particle 2 */}
      <div className="absolute top-1/3 left-1/3 w-1 h-1 bg-brand-primary/30 rounded-full blur-sm animate-particle-float-slow" style={{animationDelay: '1s'}}></div>
      
      {/* Particle 3 */}
      <div className="absolute top-2/3 right-1/4 w-1.5 h-1.5 bg-brand-bright/30 rounded-full blur-sm animate-particle-float" style={{animationDelay: '2s'}}></div>
      
      {/* Particle 4 */}
      <div className="absolute bottom-1/3 left-2/3 w-1 h-1 bg-brand-primary/25 rounded-full blur-sm animate-particle-float-slow" style={{animationDelay: '1.5s'}}></div>
      
      {/* Particle 5 */}
      <div className="absolute top-1/2 right-1/3 w-1.5 h-1.5 bg-brand-bright/25 rounded-full blur-sm animate-particle-float" style={{animationDelay: '0.5s'}}></div>
    </div>

    {/* Animated accent lights */}
    <div className="absolute top-1/4 right-1/4 h-64 w-64 rounded-full bg-brand-bright/5 blur-3xl animate-glow-drift-slow pointer-events-none" style={{animationDelay: '3s'}}></div>
    <div className="absolute bottom-1/4 left-1/3 h-80 w-80 rounded-full bg-brand-primary/5 blur-3xl animate-pulse-glow-slow pointer-events-none" style={{animationDelay: '1.5s'}}></div>
    
    {/* Radial pulse accents - New Animation */}
    <div className="absolute top-1/3 right-1/6 w-3 h-3 rounded-full bg-brand-bright/20 pointer-events-none animate-radial-pulse" style={{animationDelay: '0s'}}></div>
    <div className="absolute bottom-1/3 left-1/4 w-2 h-2 rounded-full bg-brand-primary/20 pointer-events-none animate-radial-pulse-slow" style={{animationDelay: '0.5s'}}></div>
    <div className="absolute top-2/3 right-1/3 w-2 h-2 rounded-full bg-brand-bright/15 pointer-events-none animate-radial-pulse" style={{animationDelay: '1s'}}></div>

    {/* Main Content */}
    <div className="relative z-10 h-full min-h-screen w-full flex flex-col items-center justify-center px-6 py-20">
      {/* Center Content - Glassmorphism Container */}
      <div className="w-full max-w-2xl">
        {/* Badge */}
        <div className="flex justify-center mb-8 animate-fade-in">
          <div className="inline-flex items-center gap-2 rounded-full glass-effect-sm px-4 py-2 text-xs uppercase tracking-widest text-brand-bright animate-pulse-glow">
            <Sparkles className="h-4 w-4" />
            <span>AI Detection Suite</span>
          </div>
        </div>

        {/* Glass Container */}
        <div className="glass-effect rounded-3xl px-8 py-12 sm:px-12 sm:py-16 lg:px-16 lg:py-20 animate-fade-in text-center">
          {/* Hero Content */}
          <div className="space-y-6">
            {/* Title */}
            <div>
              <h1 className="font-orbitron text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight">
                <span className="gradient-text">Decepta</span>
              </h1>
              <p className="text-xl sm:text-2xl text-brand-subtle mt-4 font-light">
                Deepfake Detection Reimagined
              </p>
            </div>

            {/* Description */}
            <p className="text-base sm:text-lg text-brand-muted leading-relaxed mx-auto pt-4">
              Advanced AI-powered platform for detecting and analyzing deepfake content with precision. Upload, analyze, and gain insights into media authenticity.
            </p>

            {/* CTA Button */}
            <div className="flex justify-center pt-4">
              <Link to="/login">
                <Button 
                  leftIcon={<ArrowRight className="h-4 w-4" />}
                  className="glow-effect text-base px-6 py-3 font-semibold hover:scale-105 transition-transform duration-300"
                >
                  Get Started
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
)
