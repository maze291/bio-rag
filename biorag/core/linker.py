"""
BioRAG Entity Linker
Named Entity Recognition and linking to biomedical databases
"""

import re
import spacy
import scispacy
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import yaml
from urllib.parse import quote
import logging
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)

# Embedded entity knowledge base (subset for demo)
ENTITY_KB_YAML = """
# Gene/Protein entities
BRCA1:
  type: gene
  name: "Breast Cancer 1"
  uniprot_id: "P38398"
  ncbi_gene_id: "672"
  aliases: ["BRCAI", "BRCC1", "BROVCA1", "IRIS", "PNCA4", "PPP1R53", "PSCP", "RNF53"]

BRCA2:
  type: gene
  name: "Breast Cancer 2"
  uniprot_id: "P51587"
  ncbi_gene_id: "675"
  aliases: ["BRCC2", "BROVCA2", "FACD", "FAD", "FAD1", "FANCD1", "PNCA2"]

TP53:
  type: gene
  name: "Tumor Protein P53"
  uniprot_id: "P04637"
  ncbi_gene_id: "7157"
  aliases: ["p53", "LFS1", "TRP53"]

EGFR:
  type: gene
  name: "Epidermal Growth Factor Receptor"
  uniprot_id: "P00533"
  ncbi_gene_id: "1956"
  aliases: ["ERBB", "ERBB1", "HER1", "mENA", "NISBD2", "PIG61"]

KRAS:
  type: gene
  name: "KRAS Proto-Oncogene"
  uniprot_id: "P01116"
  ncbi_gene_id: "3845"
  aliases: ["KRAS2", "RASK2", "NS", "NS3", "CFC2", "RALD", "K-RAS", "KI-RAS", "KRAS1"]

VEGFA:
  type: gene
  name: "Vascular Endothelial Growth Factor A"
  uniprot_id: "P15692"
  ncbi_gene_id: "7422"
  aliases: ["VEGF", "VPF", "VEGF-A", "MVCD1"]

HER2:
  type: gene
  name: "Human Epidermal Growth Factor Receptor 2"
  uniprot_id: "P04626"
  ncbi_gene_id: "2064"
  aliases: ["ERBB2", "NEU", "NGL", "HER-2", "TKR1", "HER-2/neu"]

# Chemical/Drug entities
tamoxifen:
  type: chemical
  name: "Tamoxifen"
  pubchem_cid: "2733526"
  chembl_id: "CHEMBL83"
  drugbank_id: "DB00675"
  aliases: ["Nolvadex", "Soltamox", "TAM"]

doxorubicin:
  type: chemical
  name: "Doxorubicin"
  pubchem_cid: "31703"
  chembl_id: "CHEMBL53"
  drugbank_id: "DB00997"
  aliases: ["Adriamycin", "DOX", "hydroxydaunorubicin", "Adriablastin"]

imatinib:
  type: chemical
  name: "Imatinib"
  pubchem_cid: "5291"
  chembl_id: "CHEMBL941"
  drugbank_id: "DB00619"
  aliases: ["Gleevec", "Glivec", "STI571", "CGP57148B"]

trastuzumab:
  type: chemical
  name: "Trastuzumab"
  pubchem_cid: "110635003"
  drugbank_id: "DB00072"
  aliases: ["Herceptin", "anti-HER2", "rhuMAb HER2"]

# Disease entities
breast_cancer:
  type: disease
  name: "Breast Cancer"
  mesh_id: "D001943"
  icd10: "C50"
  omim_id: "114480"
  aliases: ["breast carcinoma", "mammary cancer", "breast neoplasm"]

lung_cancer:
  type: disease
  name: "Lung Cancer"
  mesh_id: "D008175"
  icd10: "C34"
  omim_id: "211980"
  aliases: ["lung carcinoma", "pulmonary cancer", "lung neoplasm"]

leukemia:
  type: disease
  name: "Leukemia"
  mesh_id: "D007938"
  icd10: "C95"
  aliases: ["leukaemia", "blood cancer", "leucaemia"]

melanoma:
  type: disease
  name: "Melanoma"
  mesh_id: "D008545"
  icd10: "C43"
  omim_id: "155600"
  aliases: ["malignant melanoma", "cutaneous melanoma"]

# Pathway/Process entities
apoptosis:
  type: process
  name: "Apoptosis"
  go_id: "GO:0006915"
  aliases: ["programmed cell death", "PCD", "cell death"]

angiogenesis:
  type: process
  name: "Angiogenesis"
  go_id: "GO:0001525"
  aliases: ["neovascularization", "vascular development", "blood vessel formation"]

metastasis:
  type: process
  name: "Metastasis"
  go_id: "GO:0007165"
  aliases: ["cancer spread", "tumor dissemination", "secondary tumor"]
"""


@dataclass
class Entity:
    """Represents a recognized biomedical entity"""
    text: str
    label: str
    start: int
    end: int
    kb_id: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


class EntityLinker:
    """
    Biomedical entity recognition and linking system
    """

    # Class variable for model caching
    _loaded_models = {}

    def __init__(self):
        """Initialize entity linker with SciSpacy models"""
        # Load knowledge base
        self.entity_kb = yaml.safe_load(ENTITY_KB_YAML)

        # Build reverse lookup for aliases
        self._build_alias_lookup()

        # Load models (lazy loading)
        self.nlp_bio = None
        self.nlp_chem = None
        self._models_loaded = False

        # URL templates for external databases
        self.url_templates = {
            'uniprot': 'https://www.uniprot.org/uniprotkb/{id}',
            'ncbi_gene': 'https://www.ncbi.nlm.nih.gov/gene/{id}',
            'pubchem': 'https://pubchem.ncbi.nlm.nih.gov/compound/{id}',
            'chembl': 'https://www.ebi.ac.uk/chembl/compound_report_card/{id}',
            'drugbank': 'https://go.drugbank.com/drugs/{id}',
            'mesh': 'https://meshb.nlm.nih.gov/record/ui?ui={id}',
            'omim': 'https://omim.org/entry/{id}',
            'go': 'http://amigo.geneontology.org/amigo/term/{id}'
        }

        # Label mapping for different model outputs
        self.label_mapping = {
            # Standard labels
            'GENE': 'gene',
            'PROTEIN': 'protein',
            'CHEMICAL': 'chemical',
            'DISEASE': 'disease',
            'DRUG': 'chemical',
            # SciSpacy specific labels
            'GENE_OR_GENE_PRODUCT': 'gene',
            'ORGANISM': 'organism',
            'SIMPLE_CHEMICAL': 'chemical',
            'DISEASE_OR_SYNDROME': 'disease',
            'CELL_TYPE': 'cell',
            'CELL_LINE': 'cell',
            'ORGAN_OR_TISSUE': 'tissue',
            'CANCER': 'disease',
            'ANATOMICAL_SYSTEM': 'anatomy',
            'ORGANISM_SUBSTANCE': 'substance',
            'DEVELOPING_ANATOMICAL_STRUCTURE': 'anatomy',
            'PATHOLOGICAL_FORMATION': 'disease',
            'ORGANISM_SUBDIVISION': 'anatomy',
            'TISSUE': 'tissue',
            'CELL': 'cell',
            'CELLULAR_COMPONENT': 'cell_component',
            'IMMATERIAL_ANATOMICAL_ENTITY': 'anatomy',
            'MULTI_TISSUE_STRUCTURE': 'tissue',
            'ORGANISM_ATTRIBUTE': 'attribute',
            'PHENOTYPE': 'phenotype',
            'BIOLOGICAL_PROCESS': 'process'
        }

    def _load_models(self):
        """Load SciSpacy NER models with caching"""
        if self._models_loaded:
            return

        # Suppress spacy warnings about model compatibility
        warnings.filterwarnings("ignore", message=".*W095.*")

        try:
            # Check class-level cache first
            if 'bio' in self._loaded_models:
                self.nlp_bio = self._loaded_models['bio']
                logger.info("Using cached biomedical model")
            else:
                # Try to load biomedical NER models
                try:
                    # Model for general biomedical entities
                    self.nlp_bio = spacy.load("en_ner_bionlp13cg_md")
                    logger.info("Loaded en_ner_bionlp13cg_md model")
                except:
                    logger.warning("Could not load en_ner_bionlp13cg_md, trying alternatives")
                    try:
                        self.nlp_bio = spacy.load("en_core_sci_sm")
                        logger.info("Loaded en_core_sci_sm model")
                    except:
                        # Fallback to basic English model
                        try:
                            self.nlp_bio = spacy.load("en_core_web_sm")
                            logger.warning("Using basic English model as fallback")
                        except:
                            raise RuntimeError("No spacy model found. Please install en_core_web_sm at minimum.")

                # Cache the loaded model
                self._loaded_models['bio'] = self.nlp_bio

            # Try to load disease/chemical model
            if 'chem' in self._loaded_models:
                self.nlp_chem = self._loaded_models['chem']
                logger.info("Using cached chemical/disease model")
            else:
                try:
                    self.nlp_chem = spacy.load("en_ner_bc5cdr_md")
                    logger.info("Loaded en_ner_bc5cdr_md model")
                    self._loaded_models['chem'] = self.nlp_chem
                except:
                    logger.warning("Could not load en_ner_bc5cdr_md, using main model")
                    self.nlp_chem = self.nlp_bio

            self._models_loaded = True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _build_alias_lookup(self):
        """Build reverse lookup dictionary for aliases"""
        self.alias_lookup = {}

        for entity_id, info in self.entity_kb.items():
            # Add main ID
            self.alias_lookup[entity_id.lower()] = entity_id

            # Add aliases
            if 'aliases' in info:
                for alias in info['aliases']:
                    self.alias_lookup[alias.lower()] = entity_id

            # Add name
            if 'name' in info:
                self.alias_lookup[info['name'].lower()] = entity_id

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            List of Entity objects
        """
        # Load models if not already loaded
        if not self._models_loaded:
            self._load_models()

        entities = []
        seen_spans: Set[Tuple[int, int]] = set()

        # Process with both models
        doc_bio = self.nlp_bio(text)
        doc_chem = self.nlp_chem(text) if self.nlp_chem != self.nlp_bio else doc_bio

        # Extract entities from biomedical model
        for ent in doc_bio.ents:
            span = (ent.start_char, ent.end_char)
            if span not in seen_spans:
                entity = Entity(
                    text=ent.text,
                    label=self._normalize_label(ent.label_),
                    start=ent.start_char,
                    end=ent.end_char
                )
                entities.append(entity)
                seen_spans.add(span)

        # Add chemical/disease entities if using separate model
        if self.nlp_chem != self.nlp_bio:
            for ent in doc_chem.ents:
                span = (ent.start_char, ent.end_char)
                if span not in seen_spans:
                    entity = Entity(
                        text=ent.text,
                        label=self._normalize_label(ent.label_),
                        start=ent.start_char,
                        end=ent.end_char
                    )
                    entities.append(entity)
                    seen_spans.add(span)

        # Link entities to knowledge base
        for entity in entities:
            self._link_entity(entity)

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def _normalize_label(self, label: str) -> str:
        """Normalize entity labels from different models"""
        return self.label_mapping.get(label, label.lower())

    def _link_entity(self, entity: Entity):
        """
        Link entity to knowledge base and generate URL

        Args:
            entity: Entity object to link
        """
        # Normalize text for lookup
        normalized = entity.text.lower().strip()

        # Check direct match
        if normalized in self.alias_lookup:
            kb_id = self.alias_lookup[normalized]
            kb_entry = self.entity_kb[kb_id]

            entity.kb_id = kb_id
            entity.aliases = kb_entry.get('aliases', [])
            entity.description = kb_entry.get('name', entity.text)

            # Generate URL based on type
            entity.url = self._generate_url(kb_entry)
        else:
            # Try partial matching for complex entities
            for alias, kb_id in self.alias_lookup.items():
                if alias in normalized or normalized in alias:
                    kb_entry = self.entity_kb[kb_id]
                    entity.kb_id = kb_id
                    entity.aliases = kb_entry.get('aliases', [])
                    entity.description = kb_entry.get('name', entity.text)
                    entity.url = self._generate_url(kb_entry)
                    break

            # If still no match, generate generic search URL
            if not entity.url:
                entity.url = self._generate_search_url(entity)
                entity.description = entity.description or entity.text
                entity.kb_id = entity.kb_id or "N/A"

    def _generate_url(self, kb_entry: Dict[str, Any]) -> str:
        """
        Generate URL for a knowledge base entry

        Args:
            kb_entry: Knowledge base entry

        Returns:
            URL string
        """
        entity_type = kb_entry.get('type', '')

        # Gene/Protein URLs
        if entity_type == 'gene':
            if 'uniprot_id' in kb_entry:
                return self.url_templates['uniprot'].format(id=kb_entry['uniprot_id'])
            elif 'ncbi_gene_id' in kb_entry:
                return self.url_templates['ncbi_gene'].format(id=kb_entry['ncbi_gene_id'])

        # Chemical/Drug URLs
        elif entity_type == 'chemical':
            if 'pubchem_cid' in kb_entry:
                return self.url_templates['pubchem'].format(id=kb_entry['pubchem_cid'])
            elif 'drugbank_id' in kb_entry:
                return self.url_templates['drugbank'].format(id=kb_entry['drugbank_id'])

        # Disease URLs
        elif entity_type == 'disease':
            if 'mesh_id' in kb_entry:
                return self.url_templates['mesh'].format(id=kb_entry['mesh_id'])
            elif 'omim_id' in kb_entry:
                return self.url_templates['omim'].format(id=kb_entry['omim_id'])

        # Process URLs
        elif entity_type == 'process':
            if 'go_id' in kb_entry:
                return self.url_templates['go'].format(id=kb_entry['go_id'])

        # Default to search
        return f"https://www.ncbi.nlm.nih.gov/search/all/?term={quote(kb_entry.get('name', ''))}"

    def _generate_search_url(self, entity: Entity) -> str:
        """
        Generate search URL for unlinked entity

        Args:
            entity: Entity object

        Returns:
            Search URL
        """
        # Choose search engine based on entity type
        normalized_label = entity.label.lower()

        if normalized_label in ['gene', 'protein']:
            return f"https://www.ncbi.nlm.nih.gov/gene/?term={quote(entity.text)}"
        elif normalized_label in ['chemical', 'drug']:
            return f"https://pubchem.ncbi.nlm.nih.gov/search/#query={quote(entity.text)}"
        elif normalized_label in ['disease', 'cancer']:
            return f"https://meshb.nlm.nih.gov/search?searchInField=termDescriptor&searchType=exactMatch&searchMethod=FullWord&q={quote(entity.text)}"
        else:
            # Default to PubMed search
            return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(entity.text)}"

    @lru_cache(maxsize=128)
    def _get_entity_positions(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Cache entity positions to handle overlapping entities

        Args:
            text: Input text

        Returns:
            Dictionary mapping entity text to list of (start, end) positions
        """
        positions = {}

        # Extract all entities first to get their positions
        entities = self.extract_entities(text)

        for entity in entities:
            if entity.text not in positions:
                positions[entity.text] = []
            positions[entity.text].append((entity.start, entity.end))

        return positions

    def create_linked_html(self, text: str, entities: List[Entity]) -> str:
        """
        Create HTML with linked entities using proper span tracking

        Args:
            text: Original text
            entities: List of entities

        Returns:
            HTML string with linked entities
        """
        if not entities:
            return text.replace('\n', '<br>')

        # Create a list of replacements with their positions
        replacements = []

        for entity in entities:
            link = f'<a class="bio-entity" href="{entity.url}" target="_blank" title="{entity.description}">{entity.text}</a>'
            replacements.append({
                'start': entity.start,
                'end': entity.end,
                'text': entity.text,
                'replacement': link
            })

        # Sort by position (reverse to avoid offset issues)
        replacements.sort(key=lambda x: x['start'], reverse=True)

        # Track processed regions to avoid overlapping replacements
        processed_regions = []

        # Apply replacements
        html_text = text
        for replacement in replacements:
            start = replacement['start']
            end = replacement['end']

            # Check if this region overlaps with any processed region
            overlaps = False
            for proc_start, proc_end in processed_regions:
                if not (end <= proc_start or start >= proc_end):
                    overlaps = True
                    break

            if not overlaps:
                # Verify the text at this position matches
                if html_text[start:end] == replacement['text']:
                    html_text = html_text[:start] + replacement['replacement'] + html_text[end:]
                    processed_regions.append((start, end))
                else:
                    # Text mismatch, try to find the entity nearby
                    search_start = max(0, start - 10)
                    search_end = min(len(html_text), end + 10)
                    search_text = html_text[search_start:search_end]

                    if replacement['text'] in search_text:
                        # Found it, adjust positions
                        idx = search_text.index(replacement['text'])
                        actual_start = search_start + idx
                        actual_end = actual_start + len(replacement['text'])

                        html_text = (html_text[:actual_start] +
                                     replacement['replacement'] +
                                     html_text[actual_end:])
                        processed_regions.append((actual_start, actual_end))

        # Convert newlines to <br>
        html_text = html_text.replace('\n', '<br>')

        return html_text

    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, List[Dict]]:
        """
        Create summary of entities by type

        Args:
            entities: List of entities

        Returns:
            Dictionary grouped by entity type
        """
        summary = {}

        for entity in entities:
            # Determine category based on normalized label
            label = entity.label.lower()

            if label in ['gene', 'protein', 'gene_or_gene_product']:
                category = 'genes_proteins'
            elif label in ['chemical', 'drug', 'simple_chemical']:
                category = 'chemicals_drugs'
            elif label in ['disease', 'cancer', 'disease_or_syndrome']:
                category = 'diseases'
            elif label in ['cell', 'cell_type', 'cell_line']:
                category = 'cells'
            elif label in ['tissue', 'organ', 'anatomy']:
                category = 'anatomy'
            elif label in ['process', 'biological_process', 'pathway']:
                category = 'processes'
            else:
                category = 'other'

            if category not in summary:
                summary[category] = []

            # Add entity info
            summary[category].append({
                'text': entity.text,
                'type': entity.label,
                'url': entity.url,
                'description': entity.description,
                'kb_id': entity.kb_id
            })

        # Remove duplicates
        for category in summary:
            seen = set()
            unique = []
            for item in summary[category]:
                key = (item['text'], item['type'])
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            summary[category] = unique

        return summary