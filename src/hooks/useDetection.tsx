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
import type { DetectionRecord, DetectionStatus, DetectionType } from '../types'

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
  const confidence = Number((86 + Math.random() * 13).toFixed(1))
  const processingTime = Number((2.8 + Math.random() * 1.3).toFixed(2))
  return {
    id: crypto.randomUUID(),
    fileName: file.name,
    fileType: type,
    createdAt: new Date().toISOString(),
    status: 'Completed',
    confidence,
    processingTime,
    modelVersion: 'Project-AI v1.0',
    result: confidence > 82 ? 'Detected' : 'Not Detected',
    segments: Math.floor(58 + Math.random() * 132),
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
