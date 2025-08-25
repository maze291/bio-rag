# 🧬 BioRAG Frontend

A ChatGPT-style React interface for BioRAG - intelligent biomedical document analysis.

## Features

### Core Chat Interface
- **ChatGPT-like UI** - Clean, modern chat interface with message bubbles
- **Conversation Management** - Create, rename, delete, and search conversations
- **Keyboard Shortcuts** - Full keyboard navigation (⌘K, ⌘L, ⌘N, etc.)
- **Command Palette** - Quick access to all features
- **Responsive Design** - Works on desktop and mobile

### Biomedical Features
- **Entity Recognition** - Automatic detection of genes, proteins, diseases
- **Smart Entity Linking** - Clickable links to UniProt, PubChem, NCBI databases
- **Entity Panel** - Sidebar showing all detected biomedical entities
- **Jargon Simplification** - Hover tooltips for technical terms
- **Source Citations** - View supporting documents for each answer

### Advanced Settings
- **Theme Support** - Light/dark/system theme with font scaling
- **Model Selection** - Choose between BioRAG, OpenAI, or local models
- **Chat Customization** - Configure avatars, timestamps, density
- **Data Management** - Export conversations, clear history
- **BioRAG Integration** - Configure backend, enable HyDE/decomposition

## Quick Start

### Prerequisites
- Node.js 16+
- BioRAG backend running on `localhost:8000`

### Installation
```bash
cd frontend
npm install
npm start
```

The app will open at `http://localhost:3000`

### Development
```bash
npm run build    # Production build
npm test         # Run tests
npm run eject    # Eject from create-react-app (irreversible)
```

## Configuration

### Backend Connection
The frontend connects to BioRAG backend at `http://localhost:8000` by default. 

To change this:
1. Open Settings (⌘,) → BioRAG tab
2. Update "Backend URL" field
3. Or set `REACT_APP_BIORAG_URL` environment variable

### Environment Variables
Create `.env` file in frontend directory:
```bash
REACT_APP_BIORAG_URL=http://localhost:8000
REACT_APP_VERSION=1.0.0
```

## Architecture

### Components
- `App.tsx` - Main application with state management
- `Sidebar.tsx` - Conversation list and search
- `TopBar.tsx` - Navigation and model badge  
- `MessageBubble.tsx` - Chat message with biomedical enhancements
- `Composer.tsx` - Message input with file upload
- `SettingsModal.tsx` - Configuration interface
- `CommandPalette.tsx` - Quick actions (⌘K)
- `EntityPanel.tsx` - Detected entities sidebar

### API Integration
- `api/biorag.ts` - BioRAG backend communication
- Handles queries, file uploads, RSS feeds
- Graceful fallback when backend offline

### Data Storage
- Conversations stored in localStorage
- Settings persisted locally
- Export/import functionality

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Send message | `Enter` or `⌘Enter` |
| New line | `Shift+Enter` |
| Edit last message | `↑` (when input empty) |
| New conversation | `⌘N` |
| Clear conversation | `⌘L` |
| Command palette | `⌘K` |
| Settings | `⌘,` |

## Customization

### Themes
Three theme modes available:
- **System** - Follows OS preference
- **Light** - Always light mode
- **Dark** - Always dark mode

### Entity Types
Supported biomedical entities with color coding:
- 🧬 Genes (blue)
- 🧪 Proteins (green) 
- 🦠 Diseases (red)
- ⚗️ Chemicals (purple)
- 💊 Drugs (orange)
- 🔬 Cell types (pink)
- ⚙️ Biological processes (cyan)

### Message Density
- **Comfortable** - Spacious layout
- **Compact** - Dense layout for power users

## Integration with BioRAG Backend

### API Endpoints
- `POST /api/query` - Send chat message
- `POST /api/upload` - Upload documents
- `POST /api/rss` - Add RSS feed
- `GET /api/stats` - System statistics
- `POST /api/selftest` - Health check

### Response Format
```typescript
{
  enhanced_answer: string;
  entities: BioEntity[];
  source_docs: SourceDocument[];
  confidence_score?: number;
}
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit pull request

### Code Style
- TypeScript with strict mode
- React hooks and functional components
- Tailwind CSS for styling
- ESLint + Prettier formatting

## Deployment

### Build for Production
```bash
npm run build
```

### Deploy Options
- **Static hosting** - Serve `build/` folder
- **Docker** - Use provided Dockerfile
- **Netlify/Vercel** - Connect to repository

### Environment Setup
Set production environment variables:
```bash
REACT_APP_BIORAG_URL=https://your-biorag-backend.com
```

## Troubleshooting

### Backend Connection Issues
1. Check BioRAG server is running: `http://localhost:8000/health`
2. Verify CORS settings allow frontend domain
3. Check browser console for network errors

### Performance
- Conversations auto-trim after 100 messages
- Entity deduplication to prevent memory bloat  
- Lazy loading for large document sets

### Browser Support
- Modern browsers with ES6+ support
- Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

---

**Made with ❤️ for biomedical researchers**