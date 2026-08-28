export type DetectionType = 'video' | 'audio'

export type DetectionStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error'

export type DetectionResultLabel = 'DEEPFAKE (FAKE)' | 'GENUINE (REAL)'

export interface ForensicReason {
  category: string
  description: string
  location: string
  severity: 'High' | 'Medium' | 'Low' | 'Normal'
}

export interface DetectionRecord {
  id: string
  fileName: string
  fileType: DetectionType
  createdAt: string
  status: 'Completed' | 'Failed'
  confidence: number
  processingTime: number
  modelVersion: string
  result: DetectionResultLabel
  verdict: 'DEEPFAKE' | 'GENUINE'
  segments: number
  reasons?: ForensicReason[]
}

export interface AuthUser {
  id: string
  name: string
  email: string
  username: string
}

export interface SettingsState {
  appName: string
  theme: 'dark' | 'system'
  language: 'en' | 'es' | 'de'
  defaultDetectionType: DetectionType
  confidenceThreshold: number
  autoProcessing: boolean
  emailNotifications: boolean
  completionNotifications: boolean
}

export interface TimelineEvent {
  label: string
  start: number
  end: number
  active: boolean
}
