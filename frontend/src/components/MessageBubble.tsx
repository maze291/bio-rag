import React, { useState } from 'react';
import { Message, BioEntity } from '../types';
import { classNames, timeAgo } from '../utils';

interface MessageBubbleProps {
  message: Message;
  dense: boolean;
  showAvatar: boolean;
  showTimestamp: boolean;
  showCitations: boolean;
  showEntityLinks: boolean;
  showJargonTooltips: boolean;
}

// Enhanced markdown renderer with biomedical features
function renderBiomarkdown(
  text: string, 
  entities?: BioEntity[], 
  showEntityLinks = true, 
  showJargonTooltips = true
): JSX.Element {
  // Split into code blocks first
  const blocks = text.split(/```([\s\S]*?)```/g);
  const out: JSX.Element[] = [];

  for (let i = 0; i < blocks.length; i++) {
    if (i % 2 === 1) {
      // Code block
      out.push(
        <pre key={i} className="w-full overflow-x-auto rounded-lg bg-zinc-900/90 p-3 text-zinc-100 text-sm my-3">
          <code className="whitespace-pre font-mono">{blocks[i]}</code>
        </pre>
      );
    } else {
      // Process inline markdown
      const content = blocks[i];
      
      // Split text and process entities/jargon as React components
      const parts = processTextWithEntities(content, entities, showEntityLinks, showJargonTooltips);

      out.push(
        <div key={i} className="prose prose-zinc max-w-none prose-p:my-2 dark:prose-invert">
          {parts}
        </div>
      );
    }
  }

  return <>{out}</>;
}

// Process text with entities as React components (not HTML injection)
function processTextWithEntities(
  text: string, 
  entities?: BioEntity[], 
  showEntityLinks = true, 
  showJargonTooltips = true
): React.ReactNode {
  if (!text) return null;

  // For now, render as simple markdown without entity injection to avoid HTML issues
  const processedText = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code className="px-1.5 py-0.5 bg-zinc-200/70 dark:bg-zinc-700 rounded text-sm font-mono">$1</code>')
    .replace(/\n/g, '<br/>');

  // Create a simple div with the processed content
  return (
    <div 
      dangerouslySetInnerHTML={{ __html: processedText }}
      className="biorag-content"
    />
  );
}

export default function MessageBubble({
  message,
  dense,
  showAvatar,
  showTimestamp,
  showCitations,
  showEntityLinks,
  showJargonTooltips,
}: MessageBubbleProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";
  
  const bubble = (
    <div
      className={classNames(
        "max-w-[75ch] rounded-2xl px-4 py-3 break-words",
        isUser
          ? "bg-zinc-700 text-white dark:bg-zinc-700 dark:text-white"
          : "bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100",
        dense && "py-2"
      )}
    >
      {renderBiomarkdown(
        message.enhanced_answer || message.content,
        message.entities,
        showEntityLinks,
        showJargonTooltips
      )}
      
      {/* Entity count and citations for assistant messages */}
      {!isUser && (
        <div className="mt-3 flex items-center gap-3 text-[11px] opacity-70">
          {message.entities && message.entities.length > 0 && (
            <span className="flex items-center gap-1">
              🧬 {message.entities.length} entities detected
            </span>
          )}
          
          {showCitations && message.sources && message.sources.length > 0 && (
            <button 
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 hover:opacity-100 underline"
            >
              📚 {message.sources.length} sources
            </button>
          )}
        </div>
      )}

      {/* Source documents */}
      {!isUser && showSources && message.sources && (
        <div className="mt-3 space-y-2 pt-3 border-t border-zinc-200 dark:border-zinc-700">
          {message.sources.map((source, idx) => (
            <div key={idx} className="text-xs bg-zinc-50 dark:bg-zinc-800 rounded-lg p-2">
              <div className="font-medium mb-1">
                Source {idx + 1}: {source.metadata.source}
                {source.metadata.score && (
                  <span className="ml-2 text-zinc-500">
                    (relevance: {Math.round(source.metadata.score * 100)}%)
                  </span>
                )}
              </div>
              <div className="text-zinc-600 dark:text-zinc-400 line-clamp-2">
                {source.content.slice(0, 200)}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className={classNames(
      "flex w-full gap-3", 
      isUser ? "justify-end" : "justify-start"
    )}> 
      {/* Assistant Avatar */}
      {!isUser && showAvatar && (
        <div className="h-8 w-8 rounded-full bg-gradient-to-br from-emerald-400 to-blue-500 text-white grid place-items-center text-sm font-medium select-none shrink-0">
          🧬
        </div>
      )}
      
      <div className="flex flex-col items-start max-w-full">
        {bubble}
        
        {/* Timestamp */}
        {showTimestamp && (
          <div className={classNames(
            "text-[11px] text-zinc-500 mt-2 px-1",
            isUser && "text-right self-end"
          )}>
            {timeAgo(message.timestamp)}
          </div>
        )}
      </div>
      
      {/* User Avatar */}
      {isUser && showAvatar && (
        <div className="h-8 w-8 rounded-full bg-zinc-300 dark:bg-zinc-700 text-zinc-800 dark:text-zinc-100 grid place-items-center text-sm font-medium select-none shrink-0">
          U
        </div>
      )}
    </div>
  );
}