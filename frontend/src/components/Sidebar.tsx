import React, { useMemo, useState } from 'react';
import { Conversation } from '../types';
import { classNames, timeAgo } from '../utils';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: SidebarProps) {
  const [query, setQuery] = useState("");
  
  const filtered = useMemo(() => {
    if (!query) return conversations;
    return conversations.filter((c) =>
      c.title.toLowerCase().includes(query.toLowerCase())
    );
  }, [query, conversations]);

  return (
    <aside className="hidden md:flex md:w-72 shrink-0 flex-col border-r border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/40 h-full overflow-hidden">
      <div className="p-3 gap-2 flex items-center border-b border-zinc-200 dark:border-zinc-800">
        <button
          onClick={onNew}
          className="px-3 py-2 rounded-xl bg-zinc-900 text-zinc-100 dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90 transition-opacity"
          aria-label="New chat"
        >
          + New chat
        </button>
        <div className="flex-1" />
        <span className="text-sm text-zinc-500">🧬 BioRAG</span>
      </div>
      
      <div className="p-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search conversations..."
          className="w-full rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white/70 dark:bg-zinc-900 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-emerald-400 dark:focus:ring-emerald-500"
        />
      </div>
      
      <nav className="flex-1 overflow-y-auto px-2 pb-4">
        {filtered.length === 0 && (
          <div className="text-sm text-zinc-500 px-2">
            {query ? "No matching conversations" : "No conversations yet"}
          </div>
        )}
        
        {filtered.map((c) => (
          <div
            key={c.id}
            className={classNames(
              "group rounded-xl mb-1 overflow-hidden",
              c.id === activeId
                ? "bg-zinc-200 dark:bg-zinc-800"
                : "hover:bg-zinc-100 dark:hover:bg-zinc-900"
            )}
          >
            <button
              onClick={() => onSelect(c.id)}
              className="w-full text-left px-3 py-2"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="truncate text-[15px] font-medium">
                    {c.title || "Untitled"}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">
                    {timeAgo(c.updatedAt)} • {c.messages.length} messages
                  </div>
                  {c.messages.length > 0 && (
                    <div className="text-xs text-zinc-400 mt-1 truncate">
                      {c.messages[c.messages.length - 1]?.content.slice(0, 50)}...
                    </div>
                  )}
                </div>
              </div>
            </button>
            
            <div className="opacity-0 group-hover:opacity-100 transition-opacity px-2 pb-2">
              <div className="flex items-center gap-1">
                <button
                  className="text-xs px-2 py-1 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-600 dark:text-zinc-400"
                  onClick={(e) => {
                    e.stopPropagation();
                    const name = prompt("Rename chat", c.title) ?? c.title;
                    if (name !== c.title) onRename(c.id, name);
                  }}
                >
                  Rename
                </button>
                <button
                  className="text-xs px-2 py-1 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 text-red-600 dark:text-red-400"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm("Delete this conversation?")) onDelete(c.id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </nav>
      
      <div className="p-3 text-xs text-zinc-500 border-t border-zinc-200 dark:border-zinc-800">
        <div className="space-y-1">
          <p>💡 Tip: Ctrl/⌘+K for commands</p>
          <p>📄 Upload PDFs • 🔗 Add RSS feeds</p>
          <p>🧬 Smart entity linking</p>
        </div>
      </div>
    </aside>
  );
}