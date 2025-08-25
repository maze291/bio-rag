import React, { useMemo, useState, useEffect, useRef } from 'react';

interface Command {
  id: string;
  label: string;
  description?: string;
  shortcut?: string;
  category?: string;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  commands: Command[];
}

export default function CommandPalette({ 
  open, 
  onClose, 
  commands 
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands;
    
    const searchTerm = query.toLowerCase();
    return commands
      .filter(cmd => 
        cmd.label.toLowerCase().includes(searchTerm) ||
        cmd.description?.toLowerCase().includes(searchTerm) ||
        cmd.category?.toLowerCase().includes(searchTerm)
      )
      .sort((a, b) => {
        // Prioritize exact matches in label
        const aLabelMatch = a.label.toLowerCase().startsWith(searchTerm);
        const bLabelMatch = b.label.toLowerCase().startsWith(searchTerm);
        if (aLabelMatch && !bLabelMatch) return -1;
        if (!aLabelMatch && bLabelMatch) return 1;
        return a.label.localeCompare(b.label);
      });
  }, [query, commands]);

  // Reset selection when filtered commands change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands]);

  // Focus input when opened
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
      setQuery("");
      setSelectedIndex(0);
    }
  }, [open]);

  // Keyboard navigation
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(i => 
            i < filteredCommands.length - 1 ? i + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(i => 
            i > 0 ? i - 1 : filteredCommands.length - 1
          );
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].run();
            onClose();
          }
          break;
        case 'Escape':
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, filteredCommands, selectedIndex, onClose]);

  // Group commands by category
  const groupedCommands = useMemo(() => {
    if (!open) return {};
    
    const groups: Record<string, Command[]> = {};
    filteredCommands.forEach(cmd => {
      const category = cmd.category || 'General';
      if (!groups[category]) groups[category] = [];
      groups[category].push(cmd);
    });
    return groups;
  }, [filteredCommands, open]);

  if (!open) return null;

  return (
    <div 
      className="fixed inset-0 z-50 bg-black/40 p-4 flex items-start justify-center pt-[10vh]" 
      onClick={onClose}
    >
      <div 
        className="w-full max-w-2xl rounded-2xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="text-zinc-400">🔍</div>
            <input
              ref={inputRef}
              className="flex-1 bg-transparent outline-none text-lg placeholder-zinc-400"
              placeholder="Type a command or search..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="text-xs text-zinc-400 flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded text-[10px]">↑↓</kbd>
              <span>to navigate</span>
              <kbd className="px-1.5 py-0.5 bg-zinc-100 dark:bg-zinc-800 rounded text-[10px]">↵</kbd>
              <span>to select</span>
            </div>
          </div>
        </div>

        {/* Command list */}
        <div className="max-h-80 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-zinc-500">
              <div className="text-2xl mb-2">🤷‍♂️</div>
              <div>No commands found</div>
              <div className="text-xs mt-1">Try a different search term</div>
            </div>
          ) : (
            <div>
              {query.trim() === '' ? (
                // Show grouped commands when no search
                Object.entries(groupedCommands).map(([category, cmds]) => (
                  <div key={category}>
                    <div className="px-4 py-2 text-xs font-medium text-zinc-500 bg-zinc-50 dark:bg-zinc-900 sticky top-0">
                      {category}
                    </div>
                    {cmds.map((cmd, idx) => {
                      const globalIndex = filteredCommands.indexOf(cmd);
                      return (
                        <CommandItem
                          key={cmd.id}
                          command={cmd}
                          isSelected={globalIndex === selectedIndex}
                          onClick={() => {
                            cmd.run();
                            onClose();
                          }}
                        />
                      );
                    })}
                  </div>
                ))
              ) : (
                // Show flat list when searching
                filteredCommands.map((cmd, idx) => (
                  <CommandItem
                    key={cmd.id}
                    command={cmd}
                    isSelected={idx === selectedIndex}
                    onClick={() => {
                      cmd.run();
                      onClose();
                    }}
                  />
                ))
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <div>
              {filteredCommands.length} command{filteredCommands.length !== 1 ? 's' : ''} available
            </div>
            <div className="flex items-center gap-2">
              <span>Press</span>
              <kbd className="px-1.5 py-0.5 bg-zinc-200 dark:bg-zinc-700 rounded">Esc</kbd>
              <span>to close</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function CommandItem({ 
  command, 
  isSelected, 
  onClick 
}: { 
  command: Command; 
  isSelected: boolean; 
  onClick: () => void; 
}) {
  return (
    <button
      className={`w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors ${
        isSelected ? 'bg-zinc-100 dark:bg-zinc-900' : ''
      }`}
      onClick={onClick}
    >
      <div className="flex-1">
        <div className="font-medium text-sm">{command.label}</div>
        {command.description && (
          <div className="text-xs text-zinc-500 mt-0.5">{command.description}</div>
        )}
      </div>
      {command.shortcut && (
        <div className="text-xs text-zinc-400 font-mono bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
          {command.shortcut}
        </div>
      )}
    </button>
  );
}