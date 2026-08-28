import type { DetectionRecord, SettingsState } from '../types'

export const STORAGE_KEYS = {
  auth: 'depecta_auth',
  detections: 'depecta_detections',
  settings: 'depecta_settings',
  profile: 'depecta_profile',
} as const

export const INITIAL_DETECTIONS: DetectionRecord[] = [
  {
    id: 'det-001',
    fileName: 'FFPP_Sample_001.mp4',
    fileType: 'video',
    createdAt: '2026-08-27T09:14:00.000Z',
    status: 'Completed',
    confidence: 94.2,
    processingTime: 3.84,
    modelVersion: 'ResNet-50 Fine-Tuned (72.88% ROC-AUC)',
    result: 'Detected',
    segments: 128,
  },
  {
    id: 'det-002',
    fileName: 'CelebDF_Test_018.mp4',
    fileType: 'video',
    createdAt: '2026-08-26T13:26:00.000Z',
    status: 'Completed',
    confidence: 87.5,
    processingTime: 2.91,
    modelVersion: 'ResNet-50 Fine-Tuned (72.88% ROC-AUC)',
    result: 'Detected',
    segments: 74,
  },
]

export const DEFAULT_SETTINGS: SettingsState = {
  appName: 'Depecta',
  theme: 'dark',
  language: 'en',
  defaultDetectionType: 'video',
  confidenceThreshold: 70,
  autoProcessing: true,
  emailNotifications: true,
  completionNotifications: true,
}

export const formatDate = (value: string) =>
  new Date(value).toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
  })
