import { BioEntity, SourceDocument } from '../types';

export interface BioRAGResponse {
  enhanced_answer: string;
  entities: BioEntity[];
  source_docs: SourceDocument[];
  confidence_score?: number;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  document_count: number;
  documents_processed: number;
}

export interface RSSResponse {
  success: boolean;
  added: number;
  skipped: number;
  errors: string[];
  total_documents?: number;
  error?: string;
}

export class BioRAGAPI {
  private baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ||
      (import.meta as any)?.env?.VITE_API_BASE ||
      (window as any)?.__BIO_RAG_API__ ||
      'http://localhost:8000';
  }

  // Test connection to BioRAG backend
  async ping(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('BioRAG connection failed:', error);
      return false;
    }
  }

  // Send a query to BioRAG
  async query(
    message: string, 
    options: {
      enableHyDE?: boolean;
      enableDecomposition?: boolean;
      temperature?: number;
    } = {}
  ): Promise<BioRAGResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          enable_hyde: options.enableHyDE ?? true,
          enable_decomposition: options.enableDecomposition ?? true,
          temperature: options.temperature ?? 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        enhanced_answer: data.enhanced_answer || data.answer || message,
        entities: data.entities || [],
        source_docs: data.source_docs || [],
        confidence_score: data.confidence_score,
      };
    } catch (error) {
      console.error('Query error details:', {
        error,
        message: error instanceof Error ? error.message : 'Unknown error',
        baseUrl: this.baseUrl
      });
      // Return fallback response
      return {
        enhanced_answer: `**Connection Error** 🔌\n\nI can't reach the BioRAG backend server. To fix this:\n\n1. **Start the API server:**\n   \`\`\`bash\n   cd C:\\Users\\maze2\\bio-rag\n   python api_server.py\n   \`\`\`\n\n2. **Or run the Streamlit version:**\n   \`\`\`bash\n   streamlit run biorag/main.py\n   \`\`\`\n\n**Your question:** "${message}"\n\n*The server should be running at ${this.baseUrl}*`,
        entities: [],
        source_docs: [],
      };
    }
  }

  // Upload files to BioRAG
  async uploadFiles(files: File[]): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: true,
        message: data.message || 'Files uploaded successfully',
        document_count: data.document_count || 0,
        documents_processed: data.documents_processed || files.length,
      };
    } catch (error) {
      console.error('Upload error:', error);
      return {
        success: false,
        message: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        document_count: 0,
        documents_processed: 0,
      };
    }
  }

  // Add RSS feed
  async addRSSFeed(url: string): Promise<RSSResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/rss`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          rss_url: url,
        }),
      });

      if (!response.ok) {
        throw new Error(`RSS feed failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: data.success || true,
        added: data.added || 0,
        skipped: data.skipped || 0,
        errors: data.errors || [],
        total_documents: data.total_documents,
      };
    } catch (error) {
      console.error('RSS feed error:', error);
      return {
        success: false,
        added: 0,
        skipped: 0,
        errors: [error instanceof Error ? error.message : 'Unknown error'],
      };
    }
  }

  // Get system stats
  async getStats(): Promise<{
    document_count: number;
    entity_count: number;
    model_info: string;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/stats`);
      if (!response.ok) {
        throw new Error(`Stats failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        document_count: data.document_count || 0,
        entity_count: data.entity_count || 0,
        model_info: data.model_info || 'BioRAG',
      };
    } catch (error) {
      console.error('Stats error:', error);
      return {
        document_count: 0,
        entity_count: 0,
        model_info: 'BioRAG (Offline)',
      };
    }
  }

  // Clear the knowledge base
  async clearKnowledgeBase(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/clear`, {
        method: 'POST',
      });
      return response.ok;
    } catch (error) {
      console.error('Clear knowledge base error:', error);
      return false;
    }
  }

  // Run self-test
  async selfTest(): Promise<{
    success: boolean;
    message: string;
    details?: any;
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/selftest`, {
        method: 'POST',
      });

      const data = await response.json();
      return {
        success: response.ok,
        message: data.message || (response.ok ? 'Self-test passed' : 'Self-test failed'),
        details: data.details,
      };
    } catch (error) {
      console.error('Self-test error:', error);
      return {
        success: false,
        message: `Self-test failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }
}

// Default instance
export const bioragAPI = new BioRAGAPI();