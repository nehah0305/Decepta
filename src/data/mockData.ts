import type { DetectionRecord, SettingsState } from '../types'

export const STORAGE_KEYS = {
  auth: 'decepta_auth',
  detections: 'decepta_detections',
  settings: 'decepta_settings',
  profile: 'decepta_profile',
} as const

export const INITIAL_DETECTIONS: DetectionRecord[] = [
  {
    id: 'det-001',
    fileName: 'Video_001.mp4',
    fileType: 'video',
    createdAt: '2026-08-18T09:14:00.000Z',
    status: 'Completed',
    confidence: 94,
    processingTime: 3.84,
    modelVersion: 'Project-AI v1.0',
    result: 'Detected',
    segments: 128,
  },
  {
    id: 'det-002',
    fileName: 'Audio_018.wav',
    fileType: 'audio',
    createdAt: '2026-08-17T13:26:00.000Z',
    status: 'Completed',
    confidence: 87,
    processingTime: 2.91,
    modelVersion: 'Project-AI v1.0',
    result: 'Detected',
    segments: 74,
  },
]

export const DEFAULT_SETTINGS: SettingsState = {
  appName: 'Project Name',
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
