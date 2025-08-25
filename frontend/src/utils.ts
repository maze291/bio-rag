import { Settings } from './types';

export const uid = () => Math.random().toString(36).slice(2, 10);
export const now = () => Date.now();

export function classNames(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}

export function timeAgo(ts: number) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

export function saveLocal<T>(key: string, value: T) {
  try { 
    localStorage.setItem(key, JSON.stringify(value)); 
  } catch (e) {
    console.warn('Failed to save to localStorage:', e);
  }
}

export function loadLocal<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch (e) {
    console.warn('Failed to load from localStorage:', e);
    return fallback;
  }
}

export const DEFAULT_SETTINGS: Settings = {
  general: {
    language: "auto",
    messageDensity: "comfortable",
    autoSaveDrafts: true,
  },
  appearance: {
    theme: "system",
    fontScale: 1,
    codeLigatures: true,
  },
  chat: {
    showTimestamps: true,
    showAvatars: true,
    renderMarkdown: true,
    renderMath: false,
    inlineCitations: true,
    showEntityLinks: true,
    showJargonTooltips: true,
  },
  data: {
    keepHistory: true,
  },
  advanced: {
    model: "BioRAG",
    temperature: 0.7,
    enableHyDE: true,
    enableDecomposition: true,
    tools: {
      entityLinking: true,
      jargonSimplification: true,
      pdfAnalysis: true,
      rssFeeds: true,
    },
  },
  biorag: {
    backendUrl: (import.meta as any)?.env?.VITE_API_BASE || "http://localhost:8000",
    showEntityPanel: true,
    highlightEntities: true,
    showConfidenceScores: true,
  },
};