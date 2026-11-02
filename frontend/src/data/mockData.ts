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
  {
    id: 'det-003',
    fileName: 'LAVDF_Synthetic_Voice_042.wav',
    fileType: 'audio',
    createdAt: '2026-08-25T11:42:00.000Z',
    status: 'Completed',
    confidence: 97.7,
    processingTime: 0.85,
    modelVersion: 'Acoustic Transformer (97.23% ROC-AUC)',
    verdict: 'DEEPFAKE',
    result: 'DEEPFAKE (FAKE)',
    segments: 102,
    reasons: [
      {
        category: 'Cepstral Velocity Trajectory Anomaly (MFCC Δ #12)',
        location: 'Differential Spectral Envelope (Sub-band #12)',
        description: 'KernelSHAP impact +3.684. Inter-frame vocal tract movement velocity exceeds organic human articulatory physiology.',
        severity: 'High',
      },
      {
        category: 'Synthetic Vocal Tract Resonance (MFCC #16)',
        location: 'Mel-Frequency Cepstral Envelope (Coeff #16)',
        description: 'KernelSHAP impact +2.652. Timbral formant distribution closely matches neural vocoder speech synthesis filters.',
        severity: 'High',
      },
      {
        category: 'High-Frequency Spectral Energy Truncation',
        location: 'Acoustic Spectrogram (8.0 kHz – 16.0 kHz Sub-band)',
        description: 'KernelSHAP impact +1.645 on spectral rolloff. Steep high-frequency attenuation characteristic of AI vocoder upsampling.',
        severity: 'High',
      },
    ],
    acousticShap: [
      { feature: 'mfcc_delta_12', value: 3.68404 },
      { feature: 'mfcc_16', value: 2.65174 },
      { feature: 'mfcc_19', value: 2.26602 },
      { feature: 'mfcc_delta2_11', value: 1.41965 },
      { feature: 'spectral_rolloff', value: 1.15234 },
      { feature: 'mfcc_delta2_9', value: 0.93396 },
      { feature: 'mfcc_5', value: -2.39667 },
      { feature: 'f0', value: -1.82145 },
    ],
    waveform: {
      peaks: [0.65, 0.85, 1.0, 0.72, 0.66, 0.64, 0.67, 0.66, 0.65, 0.67, 0.68, 0.66, 0.65, 0.66, 0.67, 0.66],
      durationSeconds: 2.02,
      loudRegions: [
        { startSeconds: 0.01, endSeconds: 0.02, intensity: 0.924 },
        { startSeconds: 0.1, endSeconds: 0.13, intensity: 0.667 },
      ],
    },
  },
  {
    id: 'det-004',
    fileName: 'LAVDF_Genuine_Speech_018.wav',
    fileType: 'audio',
    createdAt: '2026-08-24T16:15:00.000Z',
    status: 'Completed',
    confidence: 99.7,
    processingTime: 0.78,
    modelVersion: 'Acoustic Transformer (97.23% ROC-AUC)',
    verdict: 'GENUINE',
    result: 'GENUINE (REAL)',
    segments: 151,
    reasons: [
      {
        category: 'Organic Pitch & Micro-Kinetic Modulation',
        location: 'Fundamental Frequency Track (F0 Contour over 151 frames)',
        description: 'Continuous vocal tract resonances with natural vocal fold micro-tremors and smooth biological pitch variations.',
        severity: 'Normal',
      },
      {
        category: 'Continuous Spectral Harmonic Decay',
        location: 'Acoustic Spectrogram (Full 16.0 kHz Acoustic Spectrum)',
        description: 'Smooth 1/f spectral energy decay conforming to natural vocal tract acoustics without synthetic vocoder cutoff.',
        severity: 'Normal',
      },
      {
        category: 'Organic Breath & Phonation Dynamics',
        location: 'Speech Energy Envelope (3.00s audio duration)',
        description: 'Natural respiratory breath pauses and smooth harmonic formant transitions matching real human speech.',
        severity: 'Normal',
      },
    ],
    acousticShap: [
      { feature: 'short_time_energy', value: 0.2547 },
      { feature: 'mfcc_delta_9', value: 0.07633 },
      { feature: 'mfcc_delta2_8', value: 0.05887 },
      { feature: 'mfcc_16', value: 0.04268 },
      { feature: 'mfcc_1', value: -0.02963 },
      { feature: 'spectral_flatness', value: -0.02447 },
      { feature: 'mfcc_12', value: -0.07777 },
      { feature: 'mfcc_5', value: -0.10223 },
    ],
    waveform: {
      peaks: [0.96, 0.95, 0.98, 0.96, 0.95, 0.97, 0.96, 0.92, 0.97, 0.96, 0.94, 0.97, 0.96, 0.95, 0.98, 0.96],
      durationSeconds: 3.0,
      loudRegions: [
        { startSeconds: 0.33, endSeconds: 0.35, intensity: 0.984 },
        { startSeconds: 1.75, endSeconds: 1.76, intensity: 0.979 },
      ],
    },
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
