import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import { INITIAL_DETECTIONS, STORAGE_KEYS } from '../data/mockData'
import type { DetectionRecord, DetectionStatus, DetectionType, ForensicReason } from '../types'

interface DetectionContextValue {
  detections: DetectionRecord[]
  status: DetectionStatus
  progress: number
  currentResult: DetectionRecord | null
  runDetection: (file: File, type: DetectionType) => Promise<DetectionRecord>
  getDetectionById: (id: string) => DetectionRecord | null
  resetStatus: () => void
}

const DetectionContext = createContext<DetectionContextValue | null>(null)

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

const createResult = (file: File, type: DetectionType): DetectionRecord => {
  const isExplicitFake = /fake|deepfake|manipulated|synth/i.test(file.name)
  const isExplicitReal = /real|genuine|original/i.test(file.name)
  const isFake = isExplicitFake || (!isExplicitReal && Math.random() > 0.3)

  const confidence = isFake
    ? Number((78 + Math.random() * 20).toFixed(1))
    : Number((12 + Math.random() * 22).toFixed(1))
  const processingTime = Number((2.8 + Math.random() * 1.3).toFixed(2))

  const fakeReasons: ForensicReason[] = [
    {
      category: 'Facial Boundary & Edge Blending',
      location: 'Jawline & Outer Facial Margin (Spatial RGB Domain)',
      description: 'High spatial gradient discontinuities and soft blending artifacts along the face swap boundary where the synthetic face was pasted.',
      severity: 'High',
    },
    {
      category: 'Frequency Domain Spectral Grid Noise',
      location: '2D FFT Log-Magnitude (High-Frequency Radial Sub-bands)',
      description: 'Unnatural periodic checkerboard noise patterns characteristic of neural upsampling and GAN/Diffusion face synthesis.',
      severity: 'High',
    },
    {
      category: 'Temporal Frame Jitter & Micro-Flashing',
      location: 'Frames #14 – #48 (Eye & Mouth Contour Regions)',
      description: 'Frame-to-frame feature variance exceeding human biological movement thresholds, causing micro-flashing artifacts.',
      severity: 'Medium',
    },
  ]

  const realReasons: ForensicReason[] = [
    {
      category: 'Natural Spatial Skin Texture',
      location: 'Cheekbones, Forehead & Nose Bridge (Spatial RGB Domain)',
      description: 'Continuous skin pore distribution and natural subsurface light scattering with zero edge blending anomalies.',
      severity: 'Normal',
    },
    {
      category: 'Optics Spectral Power Decay',
      location: '2D FFT Log-Magnitude Frequency Plane',
      description: 'Smooth 1/f spectral power decay characteristic of natural camera sensor optics with zero synthetic grid spikes.',
      severity: 'Normal',
    },
    {
      category: 'Bio-Kinetic Expression Synchronization',
      location: 'Full Video Sequence (Frames #01 – End)',
      description: 'Natural eye blinking velocity and smooth facial expression dynamics without spatial warping or frame drops.',
      severity: 'Normal',
    },
  ]

  return {
    id: crypto.randomUUID(),
    fileName: file.name,
    fileType: type,
    createdAt: new Date().toISOString(),
    status: 'Completed',
    confidence,
    processingTime,
    modelVersion: 'ResNet-50 Fine-Tuned (72.88% ROC-AUC)',
    verdict: isFake ? 'DEEPFAKE' : 'GENUINE',
    result: isFake ? 'DEEPFAKE (FAKE)' : 'GENUINE (REAL)',
    segments: Math.floor(58 + Math.random() * 132),
    reasons: isFake ? fakeReasons : realReasons,
  }
}

export const DetectionProvider = ({ children }: { children: ReactNode }) => {
  const [detections, setDetections] = useState<DetectionRecord[]>([])
  const [status, setStatus] = useState<DetectionStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [currentResult, setCurrentResult] = useState<DetectionRecord | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEYS.detections)
    if (stored) {
      setDetections(JSON.parse(stored) as DetectionRecord[])
      return
    }

    setDetections(INITIAL_DETECTIONS)
    localStorage.setItem(STORAGE_KEYS.detections, JSON.stringify(INITIAL_DETECTIONS))
  }, [])

  const runDetection = useCallback(async (file: File, type: DetectionType) => {
    setStatus('uploading')
    setCurrentResult(null)
    setProgress(20)
    await wait(1000)

    setStatus('processing')
    setProgress(66)
    await wait(2200)

    const result = createResult(file, type)
    setDetections((current) => {
      const nextRecords = [result, ...current]
      localStorage.setItem(STORAGE_KEYS.detections, JSON.stringify(nextRecords))
      return nextRecords
    })

    setStatus('completed')
    setProgress(100)
    setCurrentResult(result)

    return result
  }, [])

  const getDetectionById = useCallback(
    (id: string) => {
      if (id === 'latest') {
        return detections[0] ?? null
      }
      return detections.find((item) => item.id === id) ?? null
    },
    [detections],
  )

  const resetStatus = useCallback(() => {
    setStatus('idle')
    setProgress(0)
  }, [])

  const value = useMemo(
    () => ({
      detections,
      status,
      progress,
      currentResult,
      runDetection,
      getDetectionById,
      resetStatus,
    }),
    [currentResult, detections, getDetectionById, progress, resetStatus, runDetection, status],
  )

  return <DetectionContext.Provider value={value}>{children}</DetectionContext.Provider>
}

export const useDetection = () => {
  const context = useContext(DetectionContext)
  if (!context) {
    throw new Error('useDetection must be used inside DetectionProvider')
  }

  return context
}
