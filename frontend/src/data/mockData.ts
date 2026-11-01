import type { DetectionRecord, SettingsState } from '../types'

export const STORAGE_KEYS = {
  auth: 'decepta_auth',
  detections: 'decepta_detections',
  settings: 'decepta_settings',
  profile: 'decepta_profile',
} as const

// Helper to safely retrieve storage items while migrating legacy data
export const getSafeStorageItem = (key: keyof typeof STORAGE_KEYS): string | null => {
  try {
    const modernKey = STORAGE_KEYS[key]
    const modernValue = localStorage.getItem(modernKey)
    if (modernValue) return modernValue

    const legacyKey = `depecta_${key}`
    const legacyValue = localStorage.getItem(legacyKey)
    if (legacyValue) {
      localStorage.setItem(modernKey, legacyValue)
      return legacyValue
    }
  } catch {
    // LocalStorage restricted or unavailable
  }
  return null
}

export const INITIAL_DETECTIONS: DetectionRecord[] = [
  {
    id: 'det-001',
    fileName: 'DFD_Manipulated_Sequence_042.mp4',
    fileType: 'video',
    createdAt: '2026-08-27T09:14:00.000Z',
    status: 'Completed',
    confidence: 96.8,
    processingTime: 1.14,
    modelVersion: 'ResNet-50 Stage B (95.03% ROC-AUC)',
    verdict: 'DEEPFAKE',
    result: 'DEEPFAKE (FAKE)',
    segments: 128,
    reasons: [
      {
        category: 'Facial Boundary & Edge Blending',
        location: 'Jawline & Outer Facial Margin (Spatial RGB)',
        description: 'High spatial gradient discontinuities and soft blurring along the face swap boundary where the synthetic face was pasted.',
        severity: 'High',
      },
      {
        category: 'Frequency Domain Spectral Grid',
        location: '2D FFT Log-Magnitude (High-Frequency Sub-bands)',
        description: 'Unnatural periodic checkerboard noise patterns characteristic of neural upsampling and GAN/Diffusion face synthesis.',
        severity: 'High',
      },
      {
        category: 'Temporal Eye & Mouth Jitter',
        location: 'Frames #16 – #42 (Mouth & Eyebrows)',
        description: 'Inter-frame feature variance exceeding human biological movement thresholds, causing micro-flashing artifacts.',
        severity: 'Medium',
      },
    ],
  },
  {
    id: 'det-002',
    fileName: 'DFD_Original_Sequence_018.mp4',
    fileType: 'video',
    createdAt: '2026-08-26T13:26:00.000Z',
    status: 'Completed',
    confidence: 3.2,
    processingTime: 1.05,
    modelVersion: 'ResNet-50 Stage B (95.03% ROC-AUC)',
    verdict: 'GENUINE',
    result: 'GENUINE (REAL)',
    segments: 74,
    reasons: [
      {
        category: 'Natural Spatial Skin Texture',
        location: 'Cheekbones, Forehead & Nose Bridge',
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
        category: 'Bio-Kinetic Lip & Eye Synchronization',
        location: 'Sequential Frames #01 – #74',
        description: 'Natural eye blinking velocity and smooth facial expression dynamics without spatial warping or frame drops.',
        severity: 'Normal',
      },
    ],
  },
]

export const DEFAULT_SETTINGS: SettingsState = {
  appName: 'Decepta',
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
