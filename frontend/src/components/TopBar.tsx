import React from 'react';
import { classNames } from '../utils';

interface TopBarProps {
  model: string;
  onOpenSettings: () => void;
  onOpenCommands: () => void;
  documentCount?: number;
  fileCount?: number;
  onToggleFiles?: () => void;
}

export default function TopBar({
  model,
  onOpenSettings,
  onOpenCommands,
  documentCount = 0,
  fileCount = 0,
  onToggleFiles,
}: TopBarProps) {
  const getModelBadgeColor = (modelName: string) => {
    switch (modelName) {
      case "BioRAG + OpenAI":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "Local Ollama":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      default:
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200";
    }
  };

  return (
    <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-950/60 backdrop-blur px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="font-semibold tracking-tight text-lg">🧬 BioRAG</div>
        <div className={classNames(
          "text-xs rounded-lg border px-2 py-1 font-medium",
          getModelBadgeColor(model)
        )}>
          {model}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {documentCount > 0 && (
          <div className="text-sm text-zinc-500">
            📚 {documentCount} documents indexed
          </div>
        )}

        {fileCount > 0 && onToggleFiles && (
          <button
            onClick={onToggleFiles}
            className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors flex items-center gap-1"
            title="View uploaded files"
          >
            📁 {fileCount} files
          </button>
        )}
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={onOpenCommands}
          className="px-2 py-1 text-sm rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 font-mono transition-colors"
          title="Command palette (Ctrl/⌘+K)"
        >
          ⌘K
        </button>

        <button
          onClick={onOpenSettings}
          className="px-2 py-1 text-sm rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
          title="Settings (Ctrl/⌘+,)"
          aria-label="Settings"
        >
          ⚙️
        </button>
      </div>
    </header>
  );
}