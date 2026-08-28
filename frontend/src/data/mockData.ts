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
    fileName: 'FFPP_Manipulated_Sample_001.mp4',
    fileType: 'video',
    createdAt: '2026-08-27T09:14:00.000Z',
    status: 'Completed',
    confidence: 94.2,
    processingTime: 3.84,
    modelVersion: 'ResNet-50 Fine-Tuned (72.88% ROC-AUC)',
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
    fileName: 'Genuine_Interview_Sample_018.mp4',
    fileType: 'video',
    createdAt: '2026-08-26T13:26:00.000Z',
    status: 'Completed',
    confidence: 18.5,
    processingTime: 2.91,
    modelVersion: 'ResNet-50 Fine-Tuned (72.88% ROC-AUC)',
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
