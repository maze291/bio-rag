import React, { useEffect, useMemo, useRef, useState } from "react";
import { Conversation, Message, Settings, Role } from './types';
import { classNames, uid, now, saveLocal, loadLocal, DEFAULT_SETTINGS } from './utils';
import { bioragAPI } from './api/biorag';

// Components
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import MessageBubble from './components/MessageBubble';
import Composer from './components/Composer';
import SettingsModal from './components/SettingsModal';
import CommandPalette from './components/CommandPalette';
import EntityPanel from './components/EntityPanel';
import FilesPanel from './components/FilesPanel';

// Seed data
const SEED_MESSAGES: Message[] = [
  {
    id: uid(),
    role: Role.assistant,
    content: "Welcome to **BioRAG**! 🧬\n\nI'm your intelligent biomedical research assistant. I can help you:\n\n• Analyze scientific papers and documents\n• Identify genes, proteins, and biomedical entities\n• Explain complex biological processes\n• Link terms to major databases (UniProt, PubChem, NCBI)\n• Simplify technical jargon\n\nUpload some PDFs or ask me anything about biomedicine!",
    timestamp: now(),
  }
];

const SEED_CONVERSATION: Conversation = {
  id: uid(),
  title: "Welcome to BioRAG",
  messages: SEED_MESSAGES,
  createdAt: now(),
  updatedAt: now(),
};

export default function App() {
  // Settings
  const [settings, setSettings] = useState<Settings>(() => 
    loadLocal<Settings>("biorag-settings", DEFAULT_SETTINGS)
  );

  useEffect(() => {
    saveLocal("biorag-settings", settings);
  }, [settings]);

  // Theme handling
  const themeClass = useMemo(() => {
    const sysDark = window.matchMedia?.('(prefers-color-scheme: dark)')?.matches;
    const mode = settings.appearance.theme === 'system' ? (sysDark ? 'dark' : 'light') : settings.appearance.theme;
    return mode === 'dark' ? 'dark' : '';
  }, [settings.appearance.theme]);

  // Conversations
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    loadLocal<Conversation[]>("biorag-conversations", [SEED_CONVERSATION])
  );
  const [activeId, setActiveId] = useState<string>(() => {
    const storedConversations = loadLocal<Conversation[]>("biorag-conversations", [SEED_CONVERSATION]);
    const firstId = storedConversations[0]?.id ?? SEED_CONVERSATION.id;
    console.log('Initializing activeId:', { firstId, storedConversationsLength: storedConversations.length });
    return firstId;
  });

  const activeConversation = useMemo(() => {
    let found = conversations.find(c => c.id === activeId);
    
    // If no conversation found with activeId, use the first one
    if (!found && conversations.length > 0) {
      found = conversations[0];
      setActiveId(conversations[0].id); // Update activeId to match
    }
    
    // If still no conversations, ensure we have the seed conversation
    if (!found) {
      console.log('No conversations found, using seed conversation');
      setConversations([SEED_CONVERSATION]);
      setActiveId(SEED_CONVERSATION.id);
      found = SEED_CONVERSATION;
    }
    
    console.log('activeConversation computed:', { 
      activeId, 
      conversationsLength: conversations.length, 
      found: !!found,
      foundId: found?.id,
      conversations: conversations.map(c => ({ id: c.id, title: c.title }))
    });
    
    return found;
  }, [conversations, activeId]);

  useEffect(() => {
    if (settings.data.keepHistory) {
      saveLocal("biorag-conversations", conversations);
    }
  }, [conversations, settings.data.keepHistory]);

  // Chat state
  const [draft, setDraft] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [documentCount, setDocumentCount] = useState(0);
  const listRef = useRef<HTMLDivElement | null>(null);

  // UI modals
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [entityPanelOpen, setEntityPanelOpen] = useState(false);
  const [filesPanelOpen, setFilesPanelOpen] = useState(false);

  // Files state
  const [attachedFiles, setAttachedFiles] = useState<any[]>([]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTo({ 
        top: listRef.current.scrollHeight, 
        behavior: 'smooth' 
      });
    }
  }, [activeConversation?.messages?.length]);

  // Initialize BioRAG connection
  useEffect(() => {
    bioragAPI.ping().then(connected => {
      if (connected) {
        bioragAPI.getStats().then(stats => {
          setDocumentCount(stats.document_count);
        });
      }
    });
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const isModKey = e.ctrlKey || e.metaKey;
      
      if (isModKey && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
      } else if (isModKey && e.key === ',') {
        e.preventDefault();
        setSettingsOpen(true);
      } else if (isModKey && e.key.toLowerCase() === 'l') {
        e.preventDefault();
        if (window.confirm('Clear current conversation?')) {
          clearCurrentConversation();
        }
      } else if (isModKey && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        newConversation();
      } else if (e.key === 'ArrowUp' && draft.trim() === '') {
        // Edit last user message
        const lastUserMessage = activeConversation?.messages
          ?.slice()
          .reverse()
          .find(m => m.role === Role.user);
        if (lastUserMessage) {
          setDraft(lastUserMessage.content);
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [draft, activeConversation]);

  // Helper functions
  function updateActiveConversation(updater: (c: Conversation) => Conversation) {
    setConversations(prev => 
      prev.map(c => c.id === activeConversation?.id ? updater({ ...c }) : c)
    );
  }

  function newConversation() {
    const newConvo: Conversation = {
      id: uid(),
      title: "New conversation",
      messages: [],
      createdAt: now(),
      updatedAt: now(),
    };
    setConversations(prev => [newConvo, ...prev]);
    setActiveId(newConvo.id);
    setDraft("");
  }

  function renameConversation(id: string, name: string) {
    setConversations(prev => 
      prev.map(c => c.id === id ? { ...c, title: name, updatedAt: now() } : c)
    );
  }

  function deleteConversation(id: string) {
    setConversations(prev => prev.filter(c => c.id !== id));
    if (id === activeId && conversations.length > 1) {
      const remaining = conversations.filter(c => c.id !== id);
      setActiveId(remaining[0]?.id || "");
    }
  }

  function clearCurrentConversation() {
    if (!activeConversation) return;
    updateActiveConversation(c => ({ ...c, messages: [], updatedAt: now() }));
  }

  function clearAllConversations() {
    setConversations([]);
    newConversation();
  }


  async function sendMessage() {
    console.log('sendMessage called', { draft, isLoading, activeConversation: !!activeConversation });
    
    const text = draft.trim();
    // Don't block on empty text since composer handles the validation
    if (!activeConversation || isLoading) {
      console.log('sendMessage blocked:', { activeConversation: !!activeConversation, isLoading });
      return;
    }
    
    // Use text if available, otherwise use a placeholder for file-only messages
    const messageContent = text || "📎 [Files uploaded]";

    const userMessage: Message = {
      id: uid(),
      role: Role.user,
      content: messageContent,
      timestamp: now(),
    };

    // Add user message immediately
    updateActiveConversation(c => ({
      ...c,
      messages: [...c.messages, userMessage],
      updatedAt: now(),
      title: c.messages.length === 0 ? messageContent.slice(0, 30) : c.title,
    }));

    setDraft("");
    setIsLoading(true);

    try {
      // Call BioRAG API - use original text or a default query for file uploads
      const queryText = text || "Please analyze the uploaded documents and provide insights.";
      console.log('Sending query to BioRAG:', queryText);
      
      const response = await bioragAPI.query(queryText, {
        enableHyDE: settings.advanced.enableHyDE,
        enableDecomposition: settings.advanced.enableDecomposition,
        temperature: settings.advanced.temperature,
      });
      
      console.log('BioRAG response:', response);

      const assistantMessage: Message = {
        id: uid(),
        role: Role.assistant,
        content: response.enhanced_answer,
        enhanced_answer: response.enhanced_answer,
        entities: response.entities,
        sources: response.source_docs,
        timestamp: now(),
      };

      updateActiveConversation(c => ({
        ...c,
        messages: [...c.messages, assistantMessage],
        updatedAt: now(),
      }));

      // Auto-open entity panel if entities detected and enabled
      if (response.entities.length > 0 && settings.biorag.showEntityPanel && !entityPanelOpen) {
        setEntityPanelOpen(true);
      }

    } catch (error) {
      console.error('Send message error:', error);
      
      const errorMessage: Message = {
        id: uid(),
        role: Role.assistant,
        content: "I'm having trouble connecting to the BioRAG backend. Please check that the server is running and try again.",
        timestamp: now(),
      };

      updateActiveConversation(c => ({
        ...c,
        messages: [...c.messages, errorMessage],
        updatedAt: now(),
      }));
    } finally {
      setIsLoading(false);
    }
  }

  function exportConversations() {
    const data = JSON.stringify(conversations, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `biorag-conversations-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Commands for command palette
  const commands = [
    {
      id: 'new-chat',
      label: 'New conversation',
      description: 'Start a fresh chat',
      category: 'Chat',
      shortcut: '⌘N',
      run: newConversation,
    },
    {
      id: 'clear-chat',
      label: 'Clear current conversation',
      description: 'Delete all messages in this chat',
      category: 'Chat',
      shortcut: '⌘L',
      run: clearCurrentConversation,
    },
    {
      id: 'rename-chat',
      label: 'Rename conversation',
      description: 'Change the current chat title',
      category: 'Chat',
      run: () => {
        const name = prompt('Rename conversation', activeConversation?.title || '');
        if (name && activeConversation) {
          renameConversation(activeConversation.id, name);
        }
      },
    },
    {
      id: 'toggle-entities',
      label: 'Toggle entity panel',
      description: 'Show/hide detected biomedical entities',
      category: 'BioRAG',
      run: () => setEntityPanelOpen(!entityPanelOpen),
    },
    {
      id: 'toggle-files',
      label: 'Toggle files panel',
      description: 'Show/hide uploaded files',
      category: 'BioRAG',
      run: () => setFilesPanelOpen(!filesPanelOpen),
    },
    {
      id: 'upload-files',
      label: 'Upload documents',
      description: 'Use the attach button in the composer',
      category: 'BioRAG',
      run: () => alert('Use the 📎 Attach button in the composer to upload files'),
    },
    {
      id: 'self-test',
      label: 'Run BioRAG self-test',
      description: 'Test connection and functionality',
      category: 'BioRAG',
      run: async () => {
        const result = await bioragAPI.selfTest();
        alert(result.success ? '✅ Self-test passed!' : '❌ Self-test failed: ' + result.message);
      },
    },
    {
      id: 'settings',
      label: 'Open settings',
      description: 'Configure appearance and behavior',
      category: 'App',
      shortcut: '⌘,',
      run: () => setSettingsOpen(true),
    },
    {
      id: 'export',
      label: 'Export conversations',
      description: 'Download chat history as JSON',
      category: 'Data',
      run: exportConversations,
    },
    {
      id: 'toggle-theme',
      label: 'Toggle theme',
      description: 'Switch between light and dark mode',
      category: 'App',
      run: () => setSettings(prev => ({
        ...prev,
        appearance: {
          ...prev.appearance,
          theme: prev.appearance.theme === 'dark' ? 'light' : 'dark',
        },
      })),
    },
  ];

  // Get all entities from current conversation
  const allEntities = useMemo(() => {
    const entities = activeConversation?.messages
      ?.flatMap(m => m.entities || []) || [];
    
    // Deduplicate by text + type
    const unique = entities.filter((entity, index, array) => 
      array.findIndex(e => e.text === entity.text && e.type === entity.type) === index
    );
    
    return unique;
  }, [activeConversation?.messages]);

  // Density and scaling
  const dense = settings.general.messageDensity === 'compact';
  const scaleStyle: React.CSSProperties = { 
    fontSize: `${settings.appearance.fontScale}rem` 
  };

  return (
    <div className={classNames(themeClass, "min-h-screen w-full")}>
      <div 
        className="bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 transition-colors min-h-screen"
        style={scaleStyle}
      >
        <div className="mx-auto max-w-[1400px] flex relative h-screen">
          {/* Sidebar */}
          <Sidebar
            conversations={conversations}
            activeId={activeId}
            onSelect={setActiveId}
            onNew={newConversation}
            onRename={renameConversation}
            onDelete={deleteConversation}
          />

          {/* Main column */}
          <main className="flex-1 h-full grid grid-rows-[auto,1fr,auto] overflow-hidden">
            <TopBar
              model={settings.advanced.model}
              onOpenSettings={() => setSettingsOpen(true)}
              onOpenCommands={() => setCommandPaletteOpen(true)}
              documentCount={documentCount}
              fileCount={attachedFiles.length}
              onToggleFiles={() => setFilesPanelOpen(!filesPanelOpen)}
            />

            {/* Message list */}
            <div ref={listRef} className="mx-auto w-full max-w-3xl px-4 pb-24 pt-6 overflow-y-auto">
              {activeConversation?.messages?.length ? (
                <div className={classNames(
                  "flex flex-col gap-4",
                  dense && "gap-2"
                )}>
                  {activeConversation.messages.map(message => (
                    <MessageBubble
                      key={message.id}
                      message={message}
                      dense={dense}
                      showAvatar={settings.chat.showAvatars}
                      showTimestamp={settings.chat.showTimestamps}
                      showCitations={settings.chat.inlineCitations}
                      showEntityLinks={settings.chat.showEntityLinks}
                      showJargonTooltips={settings.chat.showJargonTooltips}
                    />
                  ))}
                </div>
              ) : (
                <div className="pt-16 text-center text-zinc-500">
                  <div className="text-6xl mb-4">🧬</div>
                  <div className="text-xl font-medium mb-2">Welcome to BioRAG</div>
                  <div className="mb-4">Start by uploading documents or asking a question about biomedicine</div>
                  <div className="flex flex-wrap justify-center gap-2 text-sm">
                    <button 
                      onClick={() => setDraft("What is BRCA1 and how does it relate to breast cancer?")}
                      className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
                    >
                      🧬 Example Query
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Composer */}
            <Composer
              value={draft}
              setValue={setDraft}
              onSend={sendMessage}
              isLoading={isLoading}
              placeholder="Ask about genes, proteins, diseases, research papers..."
              onFilesChange={setAttachedFiles}
            />
          </main>

          {/* Entity Panel */}
          {!filesPanelOpen && (
            <EntityPanel
              entities={allEntities}
              show={entityPanelOpen && settings.biorag.showEntityPanel}
              onClose={() => setEntityPanelOpen(false)}
            />
          )}

          {/* Files Panel */}
          <FilesPanel
            files={attachedFiles}
            show={filesPanelOpen}
            onClose={() => setFilesPanelOpen(false)}
            onRemoveFile={(fileName) => {
              setAttachedFiles(prev => prev.filter(f => f.name !== fileName));
            }}
          />
        </div>

        {/* Modals */}
        <SettingsModal
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          settings={settings}
          setSettings={setSettings}
          onExport={exportConversations}
          onClear={() => {
            if (window.confirm('Delete all conversations? This cannot be undone.')) {
              clearAllConversations();
            }
          }}
        />

        <CommandPalette
          open={commandPaletteOpen}
          onClose={() => setCommandPaletteOpen(false)}
          commands={commands}
        />
      </div>
    </div>
  );
}