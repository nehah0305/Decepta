// What: This file implements the FileUploader module in frontend/src/components/ui/FileUploader.tsx.
// How: It defines the module logic, declarations, and integrations used by its callers.
// Input: Imported values, function arguments, configuration, and runtime events supplied by callers.
// Output: Exported values, returned results, rendered UI, or runtime side effects produced by this module.

import { UploadCloud, X } from 'lucide-react'
import { useRef, useState } from 'react'
import type { DetectionType } from '../../types'
import { Button } from './Button'

/**
 * What: This interface defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
interface FileUploaderProps {
  acceptType: DetectionType
  file: File | null
  onFileSelect: (file: File | null) => void
}

/**
 * What: This declaration defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const acceptedMimeMap: Record<DetectionType, string[]> = {
  video: ['video/mp4', 'video/quicktime', 'video/x-matroska', 'video/webm'],
  audio: ['audio/wav', 'audio/mpeg', 'audio/webm', 'audio/x-wav'],
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
const formatSize = (bytes: number) => {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

/**
 * What: This function defines the named behavior or data shape.
 * How: It processes the values supplied by callers according to its implementation.
 * Input: Declared parameters, props, imported values, or type members.
 * Output: The declared return value, rendered elements, exported value, or type contract.
 */
export const FileUploader = ({ acceptType, file, onFileSelect }: FileUploaderProps) => {
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const inputRef = useRef<HTMLInputElement | null>(null)
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [isOver, setIsOver] = useState(false)
  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const [error, setError] = useState('')

  /**
   * What: This declaration defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const accept = acceptType === 'video' ? 'video/*' : 'audio/*'

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const validate = (nextFile: File) => {
    /**
     * What: This function defines the named behavior or data shape.
     * How: It processes the values supplied by callers according to its implementation.
     * Input: Declared parameters, props, imported values, or type members.
     * Output: The declared return value, rendered elements, exported value, or type contract.
     */
    const isValid = acceptedMimeMap[acceptType].some((mime) => nextFile.type.includes(mime.split('/')[0]))
    if (!isValid) {
      setError(`Invalid file type. Please upload a ${acceptType} file.`)
      onFileSelect(null)
      return
    }

    setError('')
    onFileSelect(nextFile)
  }

  /**
   * What: This function defines the named behavior or data shape.
   * How: It processes the values supplied by callers according to its implementation.
   * Input: Declared parameters, props, imported values, or type members.
   * Output: The declared return value, rendered elements, exported value, or type contract.
   */
  const onDropFile = (event: React.DragEvent<HTMLButtonElement>) => {
    event.preventDefault()
    setIsOver(false)
    /**
     * What: This declaration defines the named behavior or data shape.
     * How: It processes the values supplied by callers according to its implementation.
     * Input: Declared parameters, props, imported values, or type members.
     * Output: The declared return value, rendered elements, exported value, or type contract.
     */
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
            /**
             * What: This declaration defines the named behavior or data shape.
             * How: It processes the values supplied by callers according to its implementation.
             * Input: Declared parameters, props, imported values, or type members.
             * Output: The declared return value, rendered elements, exported value, or type contract.
             */
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
