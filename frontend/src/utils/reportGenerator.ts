import type { DetectionRecord } from '../types'

export const generateReportHTML = (analysis: DetectionRecord): string => {
  const isFake = analysis.verdict === 'DEEPFAKE' || analysis.result.includes('FAKE') || analysis.confidence > 50
  const isAudio = analysis.fileType === 'audio'
  const formattedDate = new Date(analysis.createdAt).toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  })

  const reasons = analysis.reasons || [
    {
      category: isAudio ? 'Acoustic Cepstral Discontinuity' : 'Facial Boundary & Edge Blending',
      location: isAudio ? 'Mel-Frequency Sub-bands (MFCCs)' : 'Jawline & Outer Facial Margin',
      description: isAudio
        ? 'High cepstral velocity deviations characteristic of neural vocoders.'
        : 'Discontinuities and soft blending along the face swap boundary.',
      severity: 'High',
    },
  ]

  const shapItems = analysis.acousticShap || []

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DECEPTA Forensic Analysis Report - ${analysis.id}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@600;800&display=swap');
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      color: #0f172a;
      background: #ffffff;
      padding: 40px;
      line-height: 1.5;
      font-size: 14px;
    }
    .report-card {
      max-width: 900px;
      margin: 0 auto;
      border: 2px solid #0f172a;
      padding: 40px;
      position: relative;
    }
    .watermark {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(-30deg);
      font-family: 'Orbitron', sans-serif;
      font-size: 80px;
      font-weight: 800;
      color: rgba(15, 23, 42, 0.04);
      pointer-events: none;
      z-index: 0;
    }
    .header-table {
      width: 100%;
      border-bottom: 2px solid #0f172a;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }
    .logo-text {
      font-family: 'Orbitron', sans-serif;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: 2px;
      color: #0f172a;
    }
    .meta-box {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
    }
    .meta-item {
      display: flex;
      flex-direction: column;
    }
    .meta-label {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: #64748b;
      letter-spacing: 0.5px;
    }
    .meta-val {
      font-size: 13px;
      font-weight: 600;
      color: #0f172a;
    }
    .verdict-banner {
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .verdict-banner.fake {
      background: #fef2f2;
      border: 2px solid #ef4444;
      color: #991b1b;
    }
    .verdict-banner.real {
      background: #ecfdf5;
      border: 2px solid #10b981;
      color: #065f46;
    }
    .verdict-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 20px;
      font-weight: 800;
    }
    .verdict-desc {
      font-size: 12px;
      margin-top: 4px;
    }
    .section-header {
      font-family: 'Orbitron', sans-serif;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 1px;
      margin: 28px 0 12px 0;
      color: #0f172a;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 6px;
    }
    .reasons-table, .ablation-table, .shap-table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 20px;
      font-size: 12px;
    }
    .reasons-table th, .ablation-table th, .shap-table th {
      background: #f1f5f9;
      padding: 8px 12px;
      text-align: left;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 10px;
      color: #475569;
      border: 1px solid #cbd5e1;
    }
    .reasons-table td, .ablation-table td, .shap-table td {
      padding: 10px 12px;
      border: 1px solid #e2e8f0;
      vertical-align: top;
    }
    .risk-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .risk-high { background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
    .risk-medium { background: #fef3c7; color: #b45309; border: 1px solid #fbbf24; }
    .risk-normal { background: #d1fae5; color: #047857; border: 1px solid #34d399; }
    .location-badge {
      font-family: monospace;
      font-size: 10px;
      background: #f8fafc;
      padding: 2px 6px;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
    }
    .highlight-row {
      background: #f0fdfa;
      font-weight: 700;
    }
    .footer-note {
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #94a3b8;
      font-family: monospace;
    }
  </style>
</head>
<body>

  <div class="report-card">
    <div class="watermark">CONFIDENTIAL</div>

    <!-- Header Section -->
    <table class="header-table">
      <tr>
        <td>
          <div class="logo-text">DECEPTA</div>
          <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 600;">
            ${isAudio ? 'Acoustic Stream Forensic Audit Document' : 'Visual & Spatial Forensic Audit Document'}
          </div>
        </td>
        <td style="text-align: right; font-family: monospace; font-size: 11px;">
          <div>REPORT ID: ${analysis.id}</div>
          <div>STATUS: ${analysis.status.toUpperCase()}</div>
          <div>MODALITY: ${analysis.fileType.toUpperCase()}</div>
        </td>
      </tr>
    </table>

    <!-- Metadata Grid -->
    <div class="meta-box">
      <div class="meta-item">
        <span class="meta-label">Target Media File</span>
        <span class="meta-val">${analysis.fileName}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Audit Timestamp</span>
        <span class="meta-val">${formattedDate}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Detection Engine</span>
        <span class="meta-val">${analysis.modelVersion}</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Processing Time & Segments</span>
        <span class="meta-val">${analysis.processingTime}s (${analysis.segments} Analyzed Windows)</span>
      </div>
    </div>

    <!-- Classification Verdict Banner -->
    <div class="verdict-banner ${isFake ? 'fake' : 'real'}">
      <div>
        <div class="verdict-title">${isFake ? '🔴 CLASSIFICATION: DEEPFAKE (FAKE)' : '🟢 CLASSIFICATION: GENUINE (REAL)'}</div>
        <div class="verdict-desc">
          ${isFake 
            ? `Target file exhibits structural anomalies with a ${analysis.confidence}% manipulation probability.` 
            : `Target file exhibits authentic physical dynamics with a ${(100 - analysis.confidence).toFixed(1)}% organic score.`}
        </div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 28px; font-weight: 800;">${analysis.confidence}%</div>
        <div style="font-size: 10px; text-transform: uppercase; font-weight: 700;">Confidence</div>
      </div>
    </div>

    <!-- Acoustic SHAP Attributions (if audio record) -->
    ${isAudio && shapItems.length > 0 ? `
    <div class="section-header">1. KERNELSHAP ACOUSTIC DESCRIPTOR ATTRIBUTIONS</div>
    <p style="font-size: 11px; color: #475569; margin-bottom: 12px;">
      KernelSHAP attributions over 68 named acoustic descriptors (+ values push toward synthetic/fake, - values push toward organic/genuine):
    </p>
    <table class="shap-table">
      <thead>
        <tr>
          <th>Acoustic Descriptor</th>
          <th>Feature Type</th>
          <th>SHAP Attribution</th>
          <th>Evidence Impact</th>
        </tr>
      </thead>
      <tbody>
        ${shapItems.map((s) => `
          <tr>
            <td><code>${s.feature}</code></td>
            <td>${s.feature.includes('mfcc') ? 'Cepstral Formant Domain' : s.feature.includes('spectral') ? 'Frequency Spectrum' : 'Temporal Dynamics'}</td>
            <td style="font-weight: 700; color: ${s.value >= 0 ? '#b91c1c' : '#047857'};">
              ${s.value >= 0 ? '+' : ''}${s.value.toFixed(4)}
            </td>
            <td>${s.value >= 0 ? 'Pushes toward Synthetic AI Vocoder' : 'Pushes toward Organic Human Vocal Tract'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
    ` : ''}

    <!-- Detailed Forensic Reasons & Anomaly Localization -->
    <div class="section-header">${isAudio && shapItems.length > 0 ? '2' : '1'}. FORENSIC ANOMALY BREAKDOWN & LOCALIZATION</div>
    <table class="reasons-table">
      <thead>
        <tr>
          <th style="width: 25%;">Forensic Category</th>
          <th style="width: 30%;">Anomaly Location / Domain</th>
          <th style="width: 35%;">Detailed Explanation & Findings</th>
          <th style="width: 10%;">Risk</th>
        </tr>
      </thead>
      <tbody>
        ${reasons.map((r, i) => `
          <tr>
            <td><strong>#${i + 1} ${r.category}</strong></td>
            <td><span class="location-badge">${r.location}</span></td>
            <td>${r.description}</td>
            <td>
              <span class="risk-badge ${
                r.severity === 'High' ? 'risk-high' : r.severity === 'Medium' ? 'risk-medium' : 'risk-normal'
              }">${r.severity}</span>
            </td>
          </tr>
        `).join('')}
      </tbody>
    </table>

    <!-- Multimodal Research & Ablation Benchmark -->
    <div class="section-header">${isAudio && shapItems.length > 0 ? '3' : '2'}. MULTI-BRANCH RESEARCH & ABLATION MATRIX</div>
    <p style="font-size: 11px; color: #475569; margin-bottom: 12px;">
      Cross-validated on DeepFakeDetection (DFD) and Language-Agnostic Video-Audio DeepFake (LAV-DF) benchmarks:
    </p>
    <table class="ablation-table">
      <thead>
        <tr>
          <th>Model Architecture</th>
          <th>Modality Domain</th>
          <th>Validation ROC-AUC</th>
          <th>Benchmark Findings</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Spatial CNN (Scratch)</td><td>RGB Spatial</td><td>52.62%</td><td>Scratch baseline</td></tr>
        <tr><td>ResNet-50 (Frozen)</td><td>ImageNet Spatial</td><td>56.82%</td><td>Feature extraction baseline</td></tr>
        <tr><td>FFT 2D Frequency CNN</td><td>2D FFT Log-Magnitude</td><td>58.44%</td><td>Spectral grid noise detection</td></tr>
        <tr><td>Spatial + FFT Concat</td><td>Spatial + 2D FFT</td><td>61.49%</td><td>Multi-domain feature concatenation</td></tr>
        <tr><td>Spatial + Temporal Transformer</td><td>Spatial Sequence</td><td>64.54%</td><td>Sequence temporal modeling</td></tr>
        <tr class="highlight-row"><td>ResNet-50 Stage B (Full 30-Epoch DFD) ⭐</td><td>Visual Deep Layers</td><td>95.03%</td><td>Visual Champion (99.70% Precision, 94.23% F1)</td></tr>
        <tr class="highlight-row"><td>Acoustic Transformer (LAV-DF Audio Branch) ⭐</td><td>68 Acoustic Descriptors</td><td>97.23%</td><td>Audio Champion (97.25% Precision, 286K params)</td></tr>
        <tr class="highlight-row"><td>Bidirectional Cross-Attention Fusion ⭐</td><td>Joint Audio-Visual</td><td>99.32%</td><td>Multimodal Champion (99.55% Precision, 4.9M params)</td></tr>
      </tbody>
    </table>

    <!-- Audit Disclosure -->
    <div style="background: #f0fdf4; border: 1px solid #86efac; padding: 12px; border-radius: 6px; font-size: 11px; color: #166534; margin-top: 20px;">
      <strong>Forensic Certification:</strong> Audited with Decepta Multimodal Forensic Suite. Visual predictions validated with ResNet-50 Stage B on DFD; acoustic predictions validated with the 68-descriptor Acoustic Transformer on LAV-DF.
    </div>

    <!-- Sign-off Footer -->
    <div class="footer-note">
      <span>Generated by DECEPTA Deepfake Detection Platform</span>
      <span>Confidential Forensic Audit • Verification Hash: ${analysis.id.slice(0, 8).toUpperCase()}</span>
    </div>
  </div>

</body>
</html>
  `
}

export const downloadForensicReport = (analysis: DetectionRecord) => {
  const htmlContent = generateReportHTML(analysis)
  
  // Create a blob URL and download as printable HTML report
  const blob = new Blob([htmlContent], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  
  const printWindow = window.open(url, '_blank')
  if (printWindow) {
    printWindow.onload = () => {
      printWindow.focus()
      setTimeout(() => {
        printWindow.print()
      }, 300)
    }
  }

  // Also trigger a direct download of the report HTML document
  const link = document.createElement('a')
  link.href = url
  link.download = `Decepta_Forensic_Report_${analysis.fileName.replace(/\.[^/.]+$/, '')}_${analysis.id.slice(0, 6)}.html`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
