import React, { useState, useMemo } from 'react';
import { BioEntity } from '../types';
import { classNames } from '../utils';

interface EntityPanelProps {
  entities: BioEntity[];
  show: boolean;
  onClose: () => void;
}

export default function EntityPanel({ entities, show, onClose }: EntityPanelProps) {
  const [selectedType, setSelectedType] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Group entities by type
  const entityTypes = useMemo(() => {
    const types: Record<string, BioEntity[]> = {};
    entities.forEach(entity => {
      const type = entity.type || 'unknown';
      if (!types[type]) types[type] = [];
      types[type].push(entity);
    });
    return types;
  }, [entities]);

  // Filter entities
  const filteredEntities = useMemo(() => {
    let filtered = entities;
    
    if (selectedType !== 'all') {
      filtered = filtered.filter(e => e.type === selectedType);
    }
    
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(e => 
        e.text.toLowerCase().includes(query) ||
        e.description?.toLowerCase().includes(query)
      );
    }
    
    return filtered.sort((a, b) => a.text.localeCompare(b.text));
  }, [entities, selectedType, searchQuery]);

  if (!show) return null;

  const typeColors: Record<string, string> = {
    gene: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    protein: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    disease: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    chemical: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    drug: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
    'cell-type': 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200',
    'biological-process': 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200',
  };

  const getTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      gene: '🧬',
      protein: '🧪',
      disease: '🦠',
      chemical: '⚗️',
      drug: '💊',
      'cell-type': '🔬',
      'biological-process': '⚙️',
    };
    return icons[type] || '🏷️';
  };

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white dark:bg-zinc-950 border-l border-zinc-200 dark:border-zinc-800 shadow-xl overflow-hidden flex flex-col z-40">
      {/* Header */}
      <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-lg">🧬 Detected Entities</h2>
            <span className="text-sm text-zinc-500">({entities.length})</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-zinc-100 dark:hover:bg-zinc-900 rounded-lg transition-colors"
            title="Close panel"
          >
            ✕
          </button>
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search entities..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 text-sm border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900 focus:ring-2 focus:ring-emerald-500 outline-none"
        />

        {/* Type filter */}
        <div className="mt-3">
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="w-full px-3 py-2 text-sm border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-900 focus:ring-2 focus:ring-emerald-500 outline-none"
          >
            <option value="all">All Types ({entities.length})</option>
            {Object.entries(entityTypes).map(([type, ents]) => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)} ({ents.length})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Entity list */}
      <div className="flex-1 overflow-y-auto p-2">
        {filteredEntities.length === 0 ? (
          <div className="text-center text-zinc-500 py-8">
            <div className="text-2xl mb-2">🔍</div>
            <div className="text-sm">
              {searchQuery ? 'No entities match your search' : 'No entities detected yet'}
            </div>
            <div className="text-xs mt-1">
              {!searchQuery && 'Start a conversation to see biomedical entities'}
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredEntities.map((entity, index) => (
              <div
                key={`${entity.text}-${index}`}
                className="group p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 bg-white dark:bg-zinc-900/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="text-lg shrink-0">
                    {getTypeIcon(entity.type)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2 mb-1">
                      <a
                        href={entity.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-sm text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 underline decoration-1 underline-offset-2 break-words"
                      >
                        {entity.text}
                      </a>
                      <span
                        className={classNames(
                          'text-xs px-2 py-0.5 rounded-full font-medium shrink-0',
                          typeColors[entity.type] || 'bg-zinc-100 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200'
                        )}
                      >
                        {entity.type}
                      </span>
                    </div>
                    
                    {entity.description && (
                      <div className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                        {entity.description}
                      </div>
                    )}

                    {/* Action buttons */}
                    <div className="mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <div className="flex items-center gap-2">
                        <a
                          href={entity.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs px-2 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 rounded-lg hover:bg-emerald-200 dark:hover:bg-emerald-900/50 transition-colors"
                        >
                          View Database
                        </a>
                        <button
                          onClick={() => navigator.clipboard?.writeText(entity.text)}
                          className="text-xs px-2 py-1 bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
                        >
                          Copy
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer stats */}
      <div className="p-3 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50">
        <div className="text-xs text-zinc-500 space-y-1">
          <div className="flex justify-between">
            <span>Showing:</span>
            <span>{filteredEntities.length} entities</span>
          </div>
          <div className="flex justify-between">
            <span>Types:</span>
            <span>{Object.keys(entityTypes).length} categories</span>
          </div>
          {selectedType !== 'all' && (
            <div className="text-emerald-600 dark:text-emerald-400">
              Filtered by: {selectedType}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}