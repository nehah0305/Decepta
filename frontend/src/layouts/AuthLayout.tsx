import { Outlet } from 'react-router-dom'

export const AuthLayout = () => (
  <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-4" style={{
    background: 'linear-gradient(135deg, #0C111D 0%, #142633 50%, #0C111D 100%), radial-gradient(circle at 12% 16%, rgba(38, 184, 181, 0.16), transparent 40%), radial-gradient(circle at 88% 7%, rgba(80, 214, 209, 0.14), transparent 32%), radial-gradient(circle at 50% 100%, rgba(38, 184, 181, 0.08), transparent 50%)',
    backgroundAttachment: 'fixed'
  }}>
    {/* Animated floating accents */}
    <div className="pointer-events-none absolute top-1/4 right-1/4 h-64 w-64 rounded-full bg-brand-bright/5 blur-3xl animate-glow-drift-slow" style={{animationDelay: '3s'}}></div>
    <div className="pointer-events-none absolute bottom-1/4 left-1/3 h-80 w-80 rounded-full bg-brand-primary/5 blur-3xl animate-pulse-glow-slow" style={{animationDelay: '1.5s'}}></div>
    
    {/* Grid overlay */}
    <div className="pointer-events-none absolute inset-0 bg-grid-overlay opacity-20"></div>
    
    {/* Twinkling stars */}
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="absolute top-1/4 left-1/6 w-1 h-1 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '0s'}}></div>
      <div className="absolute top-1/3 right-1/5 w-0.5 h-0.5 bg-brand-bright rounded-full animate-twinkle-slow" style={{animationDelay: '0.5s'}}></div>
      <div className="absolute top-2/3 left-1/3 w-1 h-1 bg-brand-bright rounded-full animate-twinkle-slower" style={{animationDelay: '1s'}}></div>
      <div className="absolute bottom-1/3 right-1/4 w-1 h-1 bg-brand-bright rounded-full animate-twinkle" style={{animationDelay: '0.8s'}}></div>
    </div>
    
    <div className="relative w-full max-w-md animate-fade-in">
      <Outlet />
    </div>
  </div>
)
