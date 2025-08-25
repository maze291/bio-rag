export const Role = {
  user: "user",
  assistant: "assistant",
  system: "system",
} as const;

export type Message = {
  id: string;
  role: keyof typeof Role;
  content: string;
  timestamp: number;
  entities?: BioEntity[];
  sources?: SourceDocument[];
  enhanced_answer?: string;
};

export type BioEntity = {
  text: string;
  type: string;
  url: string;
  description?: string;
};

export type SourceDocument = {
  content: string;
  metadata: {
    source: string;
    score: number;
    page?: number;
  };
};

export type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

export type Settings = {
  general: {
    language: string;
    messageDensity: "comfortable" | "compact";
    autoSaveDrafts: boolean;
  };
  appearance: {
    theme: "system" | "light" | "dark";
    fontScale: 0.9 | 1 | 1.1 | 1.25;
    codeLigatures: boolean;
  };
  chat: {
    showTimestamps: boolean;
    showAvatars: boolean;
    renderMarkdown: boolean;
    renderMath: boolean;
    inlineCitations: boolean;
    showEntityLinks: boolean;
    showJargonTooltips: boolean;
  };
  data: {
    keepHistory: boolean;
  };
  advanced: {
    model: "BioRAG" | "BioRAG + OpenAI" | "Local Ollama";
    temperature: number;
    enableHyDE: boolean;
    enableDecomposition: boolean;
    tools: {
      entityLinking: boolean;
      jargonSimplification: boolean;
      pdfAnalysis: boolean;
      rssFeeds: boolean;
    };
  };
  biorag: {
    backendUrl: string;
    showEntityPanel: boolean;
    highlightEntities: boolean;
    showConfidenceScores: boolean;
  };
};