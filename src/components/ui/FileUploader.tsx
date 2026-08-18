import { UploadCloud, X } from 'lucide-react'
import { useRef, useState } from 'react'
import type { DetectionType } from '../../types'
import { Button } from './Button'

interface FileUploaderProps {
  acceptType: DetectionType
  file: File | null
  onFileSelect: (file: File | null) => void
}

const acceptedMimeMap: Record<DetectionType, string[]> = {
  video: ['video/mp4', 'video/quicktime', 'video/x-matroska', 'video/webm'],
  audio: ['audio/wav', 'audio/mpeg', 'audio/webm', 'audio/x-wav'],
}

const formatSize = (bytes: number) => {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export const FileUploader = ({ acceptType, file, onFileSelect }: FileUploaderProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isOver, setIsOver] = useState(false)
  const [error, setError] = useState('')

  const accept = acceptType === 'video' ? 'video/*' : 'audio/*'

  const validate = (nextFile: File) => {
    const isValid = acceptedMimeMap[acceptType].some((mime) => nextFile.type.includes(mime.split('/')[0]))
    if (!isValid) {
      setError(`Invalid file type. Please upload a ${acceptType} file.`)
      onFileSelect(null)
      return
    }

    setError('')
    onFileSelect(nextFile)
  }

  const onDropFile = (event: React.DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    setIsOver(false)
    const droppedFile = event.dataTransfer.files[0]
    if (droppedFile) {
      validate(droppedFile)
    }
  }

  return (
    <div className="space-y-3">
      <button
        type="button"
        onDragOver={(event) => {
          event.preventDefault()
          setIsOver(true)
        }}
        onDragLeave={() => setIsOver(false)}
        onDrop={onDropFile}
        onClick={() => inputRef.current?.click()}
        className={`group w-full rounded-2xl border border-dashed p-8 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-bright ${
          isOver ? 'border-brand-bright bg-brand-primary/10' : 'border-brand-border bg-brand-card2/30 hover:bg-brand-card2/50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="sr-only"
          onChange={(event) => {
            const selected = event.target.files?.[0]
            if (selected) {
              validate(selected)
            }
          }}
        />
        <div className="mx-auto max-w-md text-center">
          <UploadCloud className="mx-auto mb-3 h-10 w-10 text-brand-bright" />
          <p className="text-lg font-semibold text-brand-text">Upload your file</p>
          <p className="mt-2 text-sm text-brand-muted">Drag and drop or click to browse</p>
        </div>
      </button>

      {file ? (
        <div className="rounded-xl border border-brand-border bg-brand-card2/50 p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-brand-text">{file.name}</p>
              <p className="text-xs text-brand-muted">
                {file.type || 'Unknown type'} • {formatSize(file.size)}
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              className="p-2"
              aria-label="Remove selected file"
              onClick={() => {
                onFileSelect(null)
                setError('')
                if (inputRef.current) {
                  inputRef.current.value = ''
                }
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ) : null}

      {error ? <p className="text-xs text-rose-300">{error}</p> : null}
    </div>
  )
}
