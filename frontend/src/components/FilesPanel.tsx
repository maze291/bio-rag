import React from 'react';

interface AttachedFile {
  name: string;
  size: number;
  type: string;
  uploading?: boolean;
  progress?: number;
}

interface FilesPanelProps {
  files: AttachedFile[];
  show: boolean;
  onClose: () => void;
  onRemoveFile: (fileName: string) => void;
}

export default function FilesPanel({ files, show, onClose, onRemoveFile }: FilesPanelProps) {
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (!show) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-80 bg-white dark:bg-zinc-950 border-l border-zinc-200 dark:border-zinc-800 shadow-xl overflow-hidden flex flex-col z-40">
      {/* Header */}
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-lg">📁 Uploaded Files</h2>
            <span className="text-sm text-zinc-500">({files.length})</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-zinc-100 dark:hover:bg-zinc-900 rounded-lg transition-colors"
            title="Close panel"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Files list */}
      <div className="flex-1 overflow-y-auto p-4">
        {files.length === 0 ? (
          <div className="text-center text-zinc-500 py-8">
            <div className="text-2xl mb-2">📄</div>
            <div className="text-sm">No files uploaded yet</div>
            <div className="text-xs mt-1">Use the 📎 Attach button to upload files</div>
          </div>
        ) : (
          <div className="space-y-3">
            {files.map((file, index) => (
              <div
                key={index}
                className="group p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 bg-white dark:bg-zinc-900/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded flex items-center justify-center shrink-0 relative">
                    {file.uploading ? (
                      <div className="relative w-6 h-6">
                        <svg className="w-6 h-6 transform -rotate-90" viewBox="0 0 24 24">
                          <circle
                            cx="12"
                            cy="12"
                            r="10"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className="opacity-20"
                          />
                          <circle
                            cx="12"
                            cy="12"
                            r="10"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeDasharray={`${2 * Math.PI * 10}`}
                            strokeDashoffset={`${2 * Math.PI * 10 * (1 - (file.progress || 0) / 100)}`}
                            className="transition-all duration-300"
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center text-xs font-bold">
                          {file.progress || 0}%
                        </div>
                      </div>
                    ) : (
                      <span className="text-lg">
                        {file.type.includes('pdf') ? '📄' : 
                         file.type.includes('text') ? '📝' : 
                         file.type.includes('html') ? '🌐' : '📄'}
                      </span>
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-zinc-900 dark:text-zinc-100 break-words">
                      {file.name}
                    </div>
                    <div className="text-xs text-zinc-500 mt-1">
                      {file.uploading 
                        ? `${file.progress === 50 ? 'Processing...' : 'Uploading...'} ${file.progress || 0}%`
                        : formatFileSize(file.size)
                      }
                    </div>
                    
                    {/* Status indicator */}
                    <div className="mt-2">
                      {file.uploading ? (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                          {file.progress && file.progress >= 50 ? '⚙️ Processing' : '📤 Uploading'}
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                          ✅ Ready
                        </span>
                      )}
                    </div>
                  </div>

                  <button
                    onClick={() => onRemoveFile(file.name)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded text-red-500 hover:text-red-600"
                    title="Remove file"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {files.length > 0 && (
        <div className="p-3 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50">
          <div className="text-xs text-zinc-500 space-y-1">
            <div className="flex justify-between">
              <span>Total files:</span>
              <span>{files.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Total size:</span>
              <span>{formatFileSize(files.reduce((sum, f) => sum + f.size, 0))}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}