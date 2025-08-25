import React, { useState } from 'react';
import { Settings } from '../types';
import { classNames } from '../utils';

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  settings: Settings;
  setSettings: (settings: Settings) => void;
  onExport: () => void;
  onClear: () => void;
}

export default function SettingsModal({
  open,
  onClose,
  settings,
  setSettings,
  onExport,
  onClear,
}: SettingsModalProps) {
  const [tab, setTab] = useState<'General' | 'Appearance' | 'Chat' | 'Data' | 'Advanced' | 'BioRAG' | 'Shortcuts' | 'About'>('General');

  if (!open) return null;

  const tabs = ['General', 'Appearance', 'Chat', 'Data', 'Advanced', 'BioRAG', 'Shortcuts', 'About'] as const;

  function Row({ label, children, description }: { 
    label: string; 
    children: React.ReactNode; 
    description?: string;
  }) {
    return (
      <div className="flex items-start gap-4 py-3 border-b border-zinc-100 dark:border-zinc-800 last:border-0">
        <div className="w-48 flex-shrink-0">
          <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{label}</div>
          {description && (
            <div className="text-xs text-zinc-500 mt-1">{description}</div>
          )}
        </div>
        <div className="flex-1">{children}</div>
      </div>
    );
  }

  const updateSettings = (path: string[], value: any) => {
    const newSettings = JSON.parse(JSON.stringify(settings));
    let current = newSettings;
    for (let i = 0; i < path.length - 1; i++) {
      current = current[path[i]];
    }
    current[path[path.length - 1]] = value;
    setSettings(newSettings);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-4xl max-h-[90vh] rounded-2xl bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 shadow-2xl border border-zinc-200 dark:border-zinc-800 flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-6 py-4 border-b border-zinc-200 dark:border-zinc-800">
          <div className="font-semibold text-lg">⚙️ Settings</div>
          <div className="ml-auto flex gap-1 overflow-x-auto">
            {tabs.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t as any)}
                className={classNames(
                  "px-3 py-1.5 rounded-xl text-sm whitespace-nowrap transition-colors",
                  tab === t
                    ? "bg-zinc-200 dark:bg-zinc-800"
                    : "hover:bg-zinc-100 dark:hover:bg-zinc-900"
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {tab === 'General' && (
            <div>
              <Row label="Language" description="Interface language">
                <select
                  value={settings.general.language}
                  onChange={(e) => updateSettings(['general', 'language'], e.target.value)}
                  className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-transparent px-3 py-2 text-sm"
                >
                  <option value="auto">Auto-detect</option>
                  <option value="en">English</option>
                  <option value="es">Español</option>
                  <option value="fr">Français</option>
                  <option value="de">Deutsch</option>
                </select>
              </Row>
              
              <Row label="Message density" description="Spacing between messages">
                <select
                  value={settings.general.messageDensity}
                  onChange={(e) => updateSettings(['general', 'messageDensity'], e.target.value)}
                  className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-transparent px-3 py-2 text-sm"
                >
                  <option value="comfortable">Comfortable</option>
                  <option value="compact">Compact</option>
                </select>
              </Row>
              
              <Row label="Auto-save drafts" description="Save message text automatically">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.general.autoSaveDrafts}
                    onChange={(e) => updateSettings(['general', 'autoSaveDrafts'], e.target.checked)}
                    className="rounded border-zinc-300"
                  />
                  <span className="text-sm">Save drafts in localStorage</span>
                </label>
              </Row>
            </div>
          )}

          {tab === 'Appearance' && (
            <div>
              <Row label="Theme" description="Color scheme preference">
                <select
                  value={settings.appearance.theme}
                  onChange={(e) => updateSettings(['appearance', 'theme'], e.target.value)}
                  className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-transparent px-3 py-2 text-sm"
                >
                  <option value="system">System (auto)</option>
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </Row>
              
              <Row label="Font scale" description="UI text size">
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0.9}
                    max={1.25}
                    step={0.05}
                    value={settings.appearance.fontScale}
                    onChange={(e) => updateSettings(['appearance', 'fontScale'], Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm w-12">{settings.appearance.fontScale}x</span>
                </div>
              </Row>
              
              <Row label="Code ligatures" description="Typography enhancement for code">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.appearance.codeLigatures}
                    onChange={(e) => updateSettings(['appearance', 'codeLigatures'], e.target.checked)}
                  />
                  <span className="text-sm">Enable programming ligatures</span>
                </label>
              </Row>
            </div>
          )}

          {tab === 'Chat' && (
            <div>
              <Row label="Show timestamps" description="Display message times">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.chat.showTimestamps}
                    onChange={(e) => updateSettings(['chat', 'showTimestamps'], e.target.checked)}
                  />
                  <span className="text-sm">Show when messages were sent</span>
                </label>
              </Row>
              
              <Row label="Show avatars" description="User and assistant icons">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.chat.showAvatars}
                    onChange={(e) => updateSettings(['chat', 'showAvatars'], e.target.checked)}
                  />
                  <span className="text-sm">Display profile pictures</span>
                </label>
              </Row>
              
              <Row label="Render markdown" description="Format bold, italic, code">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.chat.renderMarkdown}
                    onChange={(e) => updateSettings(['chat', 'renderMarkdown'], e.target.checked)}
                  />
                  <span className="text-sm">Process **bold**, *italic*, `code`</span>
                </label>
              </Row>
              
              <Row label="Entity links" description="Biomedical term linking">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.chat.showEntityLinks}
                    onChange={(e) => updateSettings(['chat', 'showEntityLinks'], e.target.checked)}
                  />
                  <span className="text-sm">Link genes, proteins to databases</span>
                </label>
              </Row>
              
              <Row label="Jargon tooltips" description="Scientific term explanations">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.chat.showJargonTooltips}
                    onChange={(e) => updateSettings(['chat', 'showJargonTooltips'], e.target.checked)}
                  />
                  <span className="text-sm">Show definitions on hover</span>
                </label>
              </Row>
            </div>
          )}

          {tab === 'Data' && (
            <div>
              <Row label="Keep chat history" description="Store conversations locally">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.data.keepHistory}
                    onChange={(e) => updateSettings(['data', 'keepHistory'], e.target.checked)}
                  />
                  <span className="text-sm">Save to browser storage</span>
                </label>
              </Row>
              
              <Row label="Export data" description="Download conversations">
                <button
                  onClick={onExport}
                  className="px-4 py-2 text-sm rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                >
                  📄 Export as JSON
                </button>
              </Row>
              
              <Row label="Clear all data" description="Delete everything">
                <button
                  onClick={() => {
                    if (window.confirm('Delete all conversations and settings? This cannot be undone.')) {
                      onClear();
                    }
                  }}
                  className="px-4 py-2 text-sm rounded-xl bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
                >
                  🗑️ Delete all conversations
                </button>
              </Row>
            </div>
          )}

          {tab === 'Advanced' && (
            <div>
              <Row label="Model" description="AI backend selection">
                <select
                  value={settings.advanced.model}
                  onChange={(e) => updateSettings(['advanced', 'model'], e.target.value)}
                  className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-transparent px-3 py-2 text-sm"
                >
                  <option value="BioRAG">BioRAG (Local)</option>
                  <option value="BioRAG + OpenAI">BioRAG + OpenAI</option>
                  <option value="Local Ollama">Local Ollama</option>
                </select>
              </Row>
              
              <Row label="Temperature" description="Response creativity (0-2)">
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.1}
                    value={settings.advanced.temperature}
                    onChange={(e) => updateSettings(['advanced', 'temperature'], Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-sm w-12">{settings.advanced.temperature}</span>
                </div>
              </Row>
              
              <Row label="HyDE" description="Hypothetical Document Embeddings">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.advanced.enableHyDE}
                    onChange={(e) => updateSettings(['advanced', 'enableHyDE'], e.target.checked)}
                  />
                  <span className="text-sm">Improve search with synthetic docs</span>
                </label>
              </Row>
              
              <Row label="Query decomposition" description="Break complex queries">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.advanced.enableDecomposition}
                    onChange={(e) => updateSettings(['advanced', 'enableDecomposition'], e.target.checked)}
                  />
                  <span className="text-sm">Split complex questions</span>
                </label>
              </Row>
            </div>
          )}

          {tab === 'BioRAG' && (
            <div>
              <Row label="Backend URL" description="BioRAG server endpoint">
                <input
                  type="url"
                  value={settings.biorag.backendUrl}
                  onChange={(e) => updateSettings(['biorag', 'backendUrl'], e.target.value)}
                  placeholder="http://localhost:8000"
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-transparent px-3 py-2 text-sm"
                />
              </Row>
              
              <Row label="Entity panel" description="Show detected entities sidebar">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.biorag.showEntityPanel}
                    onChange={(e) => updateSettings(['biorag', 'showEntityPanel'], e.target.checked)}
                  />
                  <span className="text-sm">Display entity analysis panel</span>
                </label>
              </Row>
              
              <Row label="Highlight entities" description="Emphasize detected terms">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.biorag.highlightEntities}
                    onChange={(e) => updateSettings(['biorag', 'highlightEntities'], e.target.checked)}
                  />
                  <span className="text-sm">Color-code biological terms</span>
                </label>
              </Row>
              
              <Row label="Confidence scores" description="Show AI confidence levels">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={settings.biorag.showConfidenceScores}
                    onChange={(e) => updateSettings(['biorag', 'showConfidenceScores'], e.target.checked)}
                  />
                  <span className="text-sm">Display answer reliability</span>
                </label>
              </Row>
            </div>
          )}

          {tab === 'Shortcuts' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  ['Send message', 'Enter or Ctrl/⌘+Enter'],
                  ['New line', 'Shift+Enter'],
                  ['Edit last message', '↑ (when input empty)'],
                  ['Open settings', 'Ctrl/⌘+,'],
                  ['Command palette', 'Ctrl/⌘+K'],
                  ['Clear chat', 'Ctrl/⌘+L'],
                  ['New chat', 'Ctrl/⌘+N'],
                  ['Search conversations', 'Ctrl/⌘+F'],
                ].map(([action, shortcut]) => (
                  <div key={action} className="flex justify-between items-center p-3 rounded-lg bg-zinc-50 dark:bg-zinc-900">
                    <span className="text-sm font-medium">{action}</span>
                    <kbd className="px-2 py-1 text-xs bg-zinc-200 dark:bg-zinc-700 rounded border font-mono">
                      {shortcut}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          )}

          {tab === 'About' && (
            <div className="prose dark:prose-invert max-w-none">
              <div className="text-center mb-6">
                <div className="text-6xl mb-4">🧬</div>
                <h2 className="text-2xl font-bold mb-2">BioRAG</h2>
                <p className="text-zinc-600 dark:text-zinc-400">
                  Intelligent biomedical document analysis
                </p>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6 text-sm">
                <div>
                  <h3 className="font-semibold mb-2">Features</h3>
                  <ul className="space-y-1 text-zinc-600 dark:text-zinc-400">
                    <li>• Biomedical entity recognition</li>
                    <li>• Smart entity linking to databases</li>
                    <li>• Jargon simplification</li>
                    <li>• Multi-format document support</li>
                    <li>• Advanced retrieval (HyDE, MMR)</li>
                    <li>• Interactive chat interface</li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="font-semibold mb-2">Technology</h3>
                  <ul className="space-y-1 text-zinc-600 dark:text-zinc-400">
                    <li>• React + TypeScript</li>
                    <li>• Python backend</li>
                    <li>• SciBERT embeddings</li>
                    <li>• ChromaDB vector storage</li>
                    <li>• SciSpacy NLP</li>
                    <li>• OpenAI/Ollama LLMs</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-6 p-4 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  BioRAG is designed for researchers, students, and professionals working with 
                  biomedical literature. It combines state-of-the-art NLP with domain expertise 
                  to make complex scientific documents accessible and actionable.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-zinc-200 dark:border-zinc-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}