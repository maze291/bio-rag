import React, { useRef, useEffect, useState } from 'react';

interface ComposerProps {
  value: string;
  setValue: (value: string) => void;
  onSend: () => void;
  isLoading?: boolean;
  placeholder?: string;
  onFilesChange?: (files: AttachedFile[]) => void;
}

interface AttachedFile {
  name: string;
  size: number;
  type: string;
  uploading?: boolean;
  progress?: number;
}

export default function Composer({
  value,
  setValue,
  onSend,
  isLoading = false,
  placeholder = "Message BioRAG...",
  onFilesChange
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);

  // Notify parent of file changes
  useEffect(() => {
    onFilesChange?.(attachedFiles);
  }, [attachedFiles, onFilesChange]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  // Focus on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      onSend();
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const uploadFileWithProgress = (file: File): Promise<boolean> => {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('files', file);

      const xhr = new XMLHttpRequest();

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          // Upload progress (0-50%)
          const uploadProgress = Math.round((event.loaded / event.total) * 50);
          setAttachedFiles(prev => prev.map(f => 
            f.name === file.name ? { ...f, progress: uploadProgress } : f
          ));
        }
      };

      xhr.onload = () => {
        if (xhr.status === 200) {
          // Start processing phase (50-100%)
          setAttachedFiles(prev => prev.map(f => 
            f.name === file.name ? { ...f, progress: 50 } : f
          ));
          
          // Simulate processing time based on file size
          const processingTime = Math.min(Math.max(file.size / 50000, 2000), 10000); // 2-10 seconds
          const steps = 20;
          const stepTime = processingTime / steps;
          
          let currentStep = 0;
          const interval = setInterval(() => {
            currentStep++;
            const processingProgress = 50 + (currentStep / steps) * 50;
            
            setAttachedFiles(prev => prev.map(f => 
              f.name === file.name ? { ...f, progress: Math.round(processingProgress) } : f
            ));
            
            if (currentStep >= steps) {
              clearInterval(interval);
              setAttachedFiles(prev => prev.map(f => 
                f.name === file.name ? { ...f, uploading: false, progress: 100 } : f
              ));
              resolve(true);
            }
          }, stepTime);
          
        } else {
          let errorMessage = 'Upload failed';
          try {
            const response = JSON.parse(xhr.responseText);
            errorMessage = response.error || xhr.statusText;
          } catch (e) {
            errorMessage = xhr.statusText;
          }
          reject(new Error(errorMessage));
        }
      };

      xhr.onerror = () => {
        reject(new Error('Upload failed'));
      };

      // Use the same base URL as the bioragAPI
      const baseUrl = (import.meta as any)?.env?.VITE_API_BASE || 'http://localhost:8000';
      xhr.open('POST', `${baseUrl}/api/upload`);
      xhr.send(formData);
    });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    
    try {
      const fileArray = Array.from(files);
      
      // Show files as attached first with uploading state
      const newAttachedFiles: AttachedFile[] = fileArray.map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
        uploading: true,
        progress: 0
      }));
      setAttachedFiles(prev => [...prev, ...newAttachedFiles]);

      // Upload files one by one to track progress
      for (const file of fileArray) {
        try {
          await uploadFileWithProgress(file);
        } catch (error) {
          console.error(`Upload failed for ${file.name}:`, error);
          // Remove failed file
          setAttachedFiles(prev => prev.filter(f => f.name !== file.name));
        }
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
      e.target.value = ''; // Reset file input
    }
  };

  const removeAttachedFile = (fileName: string) => {
    setAttachedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Check if any files are currently uploading
  const hasUploadingFiles = attachedFiles.some(file => file.uploading);
  const canSend = (value.trim().length > 0 || attachedFiles.length > 0) && !isLoading && !isUploading && !hasUploadingFiles;
  

  return (
    <div className="sticky bottom-0 border-t border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-950/60 backdrop-blur">
      <div className="mx-auto max-w-3xl px-4 py-4">
        <div className="rounded-2xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 overflow-hidden shadow-sm">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            className="w-full resize-none outline-none px-4 py-3 bg-transparent text-[15px] leading-6 min-h-[54px] max-h-[200px]"
            disabled={isLoading}
          />
          
          <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-zinc-200 dark:border-zinc-800">
            <div className="flex items-center gap-2">
              <label className="cursor-pointer">
                <input
                  type="file"
                  multiple
                  accept=".pdf,.txt,.html,.md"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={isLoading || isUploading}
                />
                <span className="px-3 py-1.5 text-sm rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors flex items-center gap-1 text-zinc-600 dark:text-zinc-400">
                  {isUploading ? (
                    <>📤 Uploading...</>
                  ) : (
                    <>📎 Attach {attachedFiles.length > 0 && `(${attachedFiles.length})`}</>
                  )}
                </span>
              </label>

              <div className="text-xs text-zinc-500">
                Shift+Enter for newline
              </div>
            </div>

            <div className="flex items-center gap-2">
              {isLoading && (
                <div className="text-sm text-zinc-500 flex items-center gap-2">
                  <div className="loading-dots flex">
                    BioRAG thinking<span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              )}


              <button
                onClick={onSend}
                disabled={!canSend}
                className={`px-4 py-1.5 text-sm rounded-xl transition-all flex items-center gap-1 ${
                  canSend
                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90'
                    : 'bg-zinc-200 text-zinc-400 dark:bg-zinc-700 dark:text-zinc-500 cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <>⏸️ Stop</>
                ) : (
                  <>Send ↗️</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Quick actions */}
        <div className="flex items-center gap-2 mt-3 text-xs text-zinc-500">
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded border">Enter</kbd>
            <span>or</span>
            <kbd className="px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded border">⌘</kbd>
            <kbd className="px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded border">Enter</kbd>
            <span>to send</span>
          </div>
          <span>•</span>
          <span>PDF, TXT, HTML files supported</span>
          <span>•</span>
          <span>Max 50MB per file</span>
        </div>
      </div>
    </div>
  );
}