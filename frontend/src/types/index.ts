// What: This file implements the index module in frontend/src/types/index.ts.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export type DetectionType = 'video' | 'audio'

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export type DetectionStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error'

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export type DetectionResultLabel = 'DEEPFAKE (FAKE)' | 'GENUINE (REAL)'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export interface ForensicReason {
  category: string
  description: string
  location: string
  severity: 'High' | 'Medium' | 'Low' | 'Normal'
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export interface AcousticShapItem {
  feature: string
  value: number
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export interface AudioWaveform {
  peaks: number[]
  durationSeconds: number
  loudRegions?: Array<{
    startSeconds: number
    endSeconds: number
    intensity: number
  }>
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
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
  acousticShap?: AcousticShapItem[]
  waveform?: AudioWaveform
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export interface AuthUser {
  id: string
  name: string
  email: string
  username: string
}

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
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

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export interface TimelineEvent {
  label: string
  start: number
  end: number
  active: boolean
}
