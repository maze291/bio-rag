"""
BioRAG Glossary Manager
Handles jargon simplification and tooltip generation
"""

import re
import yaml
from typing import Dict, List, Optional, Tuple, Set
import logging
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

# Embedded glossary of biomedical terms
GLOSSARY_YAML = """
# Molecular Biology Terms
apoptosis: "programmed cell death - when cells die in a controlled way"
metastasis: "spread of cancer from one part of the body to another"
angiogenesis: "formation of new blood vessels"
oncogene: "gene that can cause cancer when mutated or overexpressed"
tumor suppressor: "gene that helps prevent uncontrolled cell growth"
mutation: "change in DNA sequence that may affect gene function"
expression: "process of using genetic information to make proteins"
transcription: "copying DNA information into RNA"
translation: "making proteins from RNA instructions"
proliferation: "rapid cell division and growth"
differentiation: "process where cells become specialized"
epigenetics: "changes in gene activity without changing DNA sequence"
methylation: "chemical modification that can turn genes on or off"
phosphorylation: "adding phosphate groups to proteins to change their activity"
pathway: "series of molecular interactions leading to cellular response"
signaling: "communication between and within cells"
receptor: "protein that receives and responds to molecular signals"
ligand: "molecule that binds to a receptor"
inhibitor: "substance that blocks or reduces activity"
agonist: "substance that activates a receptor"
antagonist: "substance that blocks a receptor"
biomarker: "measurable indicator of biological state or condition"
prognosis: "likely outcome or course of a disease"
immunotherapy: "treatment that uses the immune system to fight disease"
targeted therapy: "treatment aimed at specific molecular targets"
resistance: "when treatment stops working effectively"
remission: "reduction or disappearance of disease signs"
relapse: "return of disease after improvement"
metastatic: "relating to cancer spread"
benign: "not cancerous or harmful"
malignant: "cancerous and able to spread"
carcinoma: "cancer that starts in epithelial tissue"
sarcoma: "cancer that starts in connective tissue"
lymphoma: "cancer of the lymphatic system"
leukemia: "cancer of blood-forming tissues"
neoplasm: "abnormal growth of tissue (tumor)"
lesion: "area of abnormal tissue"
biopsy: "removal of tissue for examination"
staging: "determining extent of cancer spread"
grade: "how abnormal cancer cells look under microscope"
cytotoxic: "toxic to cells"
chemotherapy: "drug treatment for cancer"
radiation therapy: "using radiation to kill cancer cells"
adjuvant: "additional treatment after primary therapy"
neoadjuvant: "treatment given before main therapy"
palliative: "treatment to relieve symptoms without curing"
in vitro: "in laboratory conditions (test tube)"
in vivo: "in living organism"
ex vivo: "outside living organism but in natural conditions"
xenograft: "tissue graft from different species"
transgenic: "organism with genes from another species"
knockout: "organism with specific gene disabled"
wild-type: "normal, non-mutated version"
phenotype: "observable characteristics"
genotype: "genetic makeup"
allele: "variant form of a gene"
heterozygous: "having two different alleles"
homozygous: "having two identical alleles"
dominant: "allele that is expressed when present"
recessive: "allele expressed only when two copies present"
penetrance: "proportion of individuals with mutation showing symptoms"
polymorphism: "common genetic variation"
somatic: "affecting body cells, not inherited"
germline: "affecting reproductive cells, can be inherited"
chromatin: "DNA and proteins that make up chromosomes"
histone: "protein that DNA wraps around"
nucleosome: "basic unit of DNA packaging"
telomere: "protective cap at chromosome ends"
centromere: "region where sister chromatids connect"
mitosis: "cell division producing identical cells"
meiosis: "cell division producing sex cells"
cell cycle: "series of events in cell division"
checkpoint: "control point in cell cycle"
senescence: "permanent growth arrest of cells"
autophagy: "cellular self-eating for recycling"
necrosis: "uncontrolled cell death"
inflammation: "immune response to injury or infection"
cytokine: "signaling protein in immune response"
antibody: "protein that recognizes foreign substances"
antigen: "substance that triggers immune response"
epitope: "part of antigen recognized by antibody"
monoclonal: "antibodies from single cell clone"
polyclonal: "antibodies from multiple cell clones"
hybridoma: "fusion of antibody-producing and cancer cells"
immunohistochemistry: "using antibodies to detect proteins in tissue"
flow cytometry: "analyzing cells by laser"
pcr: "technique to amplify DNA"
sequencing: "determining order of DNA bases"
microarray: "tool for analyzing many genes at once"
proteomics: "study of all proteins in organism"
genomics: "study of entire genome"
transcriptomics: "study of all RNA molecules"
metabolomics: "study of metabolic products"
systems biology: "studying biological systems as whole"
bioinformatics: "using computers to analyze biological data"
biostatistics: "statistics applied to biological data"
epidemiology: "study of disease patterns in populations"
clinical trial: "research study testing treatments in people"
randomized: "participants assigned to groups by chance"
double-blind: "neither participants nor researchers know treatment"
placebo: "inactive treatment for comparison"
cohort: "group of people in study"
retrospective: "looking back at past data"
prospective: "following forward in time"
meta-analysis: "combining results from multiple studies"
p-value: "probability result occurred by chance"
confidence interval: "range of likely values"
hazard ratio: "relative risk over time"
odds ratio: "odds of outcome with vs without exposure"
sensitivity: "ability to correctly identify positive cases"
specificity: "ability to correctly identify negative cases"
false positive: "incorrectly identified as positive"
false negative: "incorrectly identified as negative"
incidence: "new cases in population over time"
prevalence: "total cases in population at given time"
morbidity: "disease or symptom rate"
mortality: "death rate"
etiology: "cause or origin of disease"
pathogenesis: "development of disease"
pathophysiology: "functional changes from disease"
homeostasis: "maintaining stable internal conditions"
metabolism: "chemical processes in living things"
catabolism: "breaking down molecules for energy"
anabolism: "building up molecules"
enzyme: "protein that speeds up reactions"
substrate: "molecule that enzyme acts on"
cofactor: "non-protein helper for enzyme"
kinase: "enzyme that adds phosphate groups"
phosphatase: "enzyme that removes phosphate groups"
protease: "enzyme that breaks down proteins"
nuclease: "enzyme that breaks down nucleic acids"
polymerase: "enzyme that builds DNA or RNA"
ligase: "enzyme that joins molecules"
oxidation: "loss of electrons"
reduction: "gain of electrons"
reactive oxygen species: "chemically reactive molecules with oxygen"
antioxidant: "substance that prevents oxidation damage"
apoptotic: "relating to programmed cell death"
necrotizing: "causing tissue death"
angiogenic: "promoting blood vessel growth"
anti-angiogenic: "preventing blood vessel growth"
proliferative: "causing rapid growth"
anti-proliferative: "preventing rapid growth"
cytostatic: "stopping cell growth"
cytocidal: "killing cells"
# Additional uppercase abbreviations
PCR: "polymerase chain reaction - technique to amplify DNA"
DNA: "deoxyribonucleic acid - molecule carrying genetic information"
RNA: "ribonucleic acid - molecule that helps express genes"
ATP: "adenosine triphosphate - energy currency of cells"
ADP: "adenosine diphosphate - lower energy form of ATP"
CRISPR: "gene editing technology using bacterial defense system"

#Cell biology & anatomy
actin: "structural protein forming microfilaments"
axon: "long nerve‑cell projection that conducts impulses"
basal_lamina: "thin extracellular layer supporting epithelium"
centriole: "cylindrical organelle involved in cell division"
chromatid: "one of two identical chromosome copies"
cilia: "hair‑like organelles moving fluid across cell surfaces"
cytosol: "liquid portion of cytoplasm"
desmosome: "junction anchoring adjacent cells together"
endosome: "membrane compartment for sorting internalised molecules"
flagellum: "whip‑like tail enabling cell motility"
gap_junction: "channel connecting cytoplasms of adjacent cells"
glycocalyx: "carbohydrate‑rich cell‑surface coat"
golgi_apparatus: "organelle modifying and packaging proteins"
hemidesmosome: "junction anchoring cells to basement membrane"
integrin: "transmembrane receptor for ECM adhesion"
intercalated_disc: "specialised junctions between cardiac muscle cells"
lysosome: "organelle containing digestive enzymes"
macropinocytosis: "bulk uptake of extracellular fluid"
peroxisome: "organelle breaking down fatty acids & ROS"
plasmodesmata: "plant cell channels connecting cytoplasm"
reticular_fibers: "thin collagen fibers in soft tissue matrix"
secretory_granule: "vesicle storing substances for exocytosis"
spindle_apparatus: "microtubule system segregating chromosomes"
stromal_cell: "connective‑tissue support cell"
teleomere: "end segment of chromosome protecting DNA"
transcytosis: "transport of molecules across a cell"

#Immunology
adjuvant_immune: "agent enhancing immune response to antigen"
allergen: "substance provoking allergic reaction"
alloantigen: "antigen differing within a species"
anaer: "immediate severe systemic allergic reaction"
antigen_presenting_cell: "cell displaying antigen with MHC"
apc: "abbr. antigen‑presenting cell"
attenuated_vaccine: "vaccine using weakened pathogen"
clonal_expansion: "proliferation of immune cell clone"
complement_cascade: "plasma protein system aiding immunity"
cytokine_storm: "excessive systemic cytokine release"
dendritic_cell: "professional antigen‑presenting immune cell"
extravasation: "leukocyte exit from bloodstream"
Fc_receptor: "surface protein binding antibody Fc"
FDC: "follicular dendritic cell"
granzyme: "serine protease from cytotoxic lymphocytes"
HLA: "human leukocyte antigen (MHC in humans)"
immunogenicity: "ability to provoke immune response"
innate_immunity: "non‑specific first‑line defense"
isohemagglutinin: "naturally occurring anti‑blood‑group antibody"
lag_phase_immunity: "delay before measurable antibody response"
lymphokine: "cytokine produced by lymphocytes"
macrophage_polarisation: "functional differentiation of macrophages"
monocytic_myeloid_derived_suppressor_cell: "immunosuppressive myeloid cell subset"
NK_cell: "natural killer lymphocyte"
opsonisation: "antibody/complement coating enhancing phagocytosis"
perforin: "pore‑forming cytotoxic protein"
seroconversion: "development of detectable antibodies"
toll_like_receptor: "pattern‑recognition receptor family"
T_reg: "regulatory T lymphocyte suppressing immunity"

#Virology & microbiology
capsid: "protein shell of a virus"
complementation_virology: "one virus supplying missing function of another"
envelope_protein: "viral membrane glycoprotein essential for entry"
lytic_cycle: "viral replication causing cell lysis"
provirus: "viral DNA integrated into host genome"
quasispecies: "diverse but related viral population"
reverse_transcriptase: "enzyme synthesising DNA from RNA"
RNA_dependent_RNA_polymerase: "viral enzyme replicating RNA genomes"
serotype: "distinct variation recognised by antibodies"
virion: "complete infectious virus particle"
virome: "entire viral community of an organism"

#Pharmacology & toxicology
bioavailability: "fraction of administered dose reaching circulation"
clearance: "volume of plasma cleared of drug per time"
cytochrome_p450: "enzyme family metabolising xenobiotics"
drug_half_life: "time to reduce concentration by half"
efficacy_drug: "maximum response achievable"
first_pass_metabolism: "drug degradation in gut/liver before circulation"
IC50: "concentration inhibiting 50 % activity"
loading_dose: "initial higher dose to reach target level"
pharmacodynamics: "drug effects on body"
pharmacogenomics: "influence of genetics on drug response"
pharmacokinetics: "body’s effect on drug"
prodrug: "inactive compound metabolised into active drug"
therapeutic_index: "ratio of toxic to effective dose"
toxicodynamics: "effects of toxins on organism"
unbound_fraction: "portion of drug not protein‑bound"

#Pathology
atrophy: "decrease in cell size or tissue mass"
dysplasia: "abnormal cell growth indicating pre‑cancer"
empyema: "pus in a body cavity"
embolus: "intravascular material blocking blood flow"
granuloma: "organised collection of macrophages"
hyperplasia: "increase in cell number"
hypertrophy: "increase in cell size"
ischemia: "insufficient blood supply"
lesion_precancerous: "abnormal tissue likely to become malignant"
necrotising_fasciitis: "rapidly spreading soft‑tissue infection"
papilloma: "benign epithelial tumor forming finger‑like projections"
paraneoplastic_syndrome: "systemic effects of cancer unrelated to mass"
scirrhous: "hard, fibrous cancer type"
steatosis: "abnormal intracellular fat accumulation"
thrombosis: "formation of blood clot within vessel"

#Genetics & genomics
aneuploidy: "abnormal chromosome number"
copy_number_variation: "segmental DNA gains or losses"
de_novo_mutation: "new mutation not inherited from parents"
exome: "all protein‑coding regions of genome"
founder_mutation: "allele common in a population from single ancestor"
frameshift_mutation: "indel altering reading frame"
GWAS: "genome‑wide association study"
haplotype: "set of alleles inherited together"
linkage_disequilibrium: "non‑random allele association"
microsatellite_instability: "tendency for short repeats to mutate"
penetrance_incomplete: "not all mutation carriers show phenotype"
recombination_hotspot: "genomic region with high crossover rate"
transversion: "purine‑pyrimidine base substitution"

#Molecular techniques
agarose_gel: "matrix for DNA electrophoresis"
allele_specific_PCR: "PCR detecting single nucleotide difference"
ChIP_seq: "chromatin immunoprecipitation sequencing"
cryo_EM: "electron microscopy under cryogenic conditions"
DNase_footprinting: "technique mapping protein‑DNA interaction sites"
elisa: "enzyme‑linked immunosorbent assay"
fluorescence_in_situ_hybridisation: "probe‑based detection of RNA/DNA in cells"
gel_shift_assay: "electrophoretic mobility shift for protein‑DNA binding"
len_single_molecule_real_time_sequencing: "PacBio long‑read method"
Mass_spectrometry_imaging: "spatial distribution of molecules via MS"
Prime_editing: "CRISPR‑based precise sequence editing"
RACE: "rapid amplification of cDNA ends"
Sanger_sequencing: "chain‑termination DNA sequencing"
scRNA_seq: "single‑cell RNA sequencing"
SMARTer_cDNA: "template‑switching cDNA synthesis"
Western_blot: "protein immunoblotting technique"

#Bioinformatics & statistics
alignment_score: "numerical measure of sequence similarity"
Bonferroni_correction: "adjusted significance threshold for multiple tests"
bootstrapping_statistics: "resampling method estimating confidence"
BUSCO: "benchmarking universal single‑copy orthologs"
contig: "continuous DNA sequence assembly fragment"
edgeR: "RNA‑seq differential expression software"
FDR: "false discovery rate"
heterozygosity_ratio: "frequency of heterozygous loci"
log_fold_change: "logarithmic ratio of expression levels"
Manhattan_plot: "scatter plot for GWAS p‑values"
Phylogenetic_tree: "diagram of evolutionary relationships"
principal_component_analysis: "dimensionality reduction technique"
q_value: "adjusted false discovery probability"
Shannon_entropy_sequence: "measurement of sequence complexity"
Volcano_plot: "scatter showing significance vs fold‑change"

#Clinical research
adverse_event: "unfavourable medical occurrence during trial"
blinding_single: "participants unaware of treatment allocation"
CONSORT_statement: "guidelines for reporting randomised trials"
co_primary_endpoint: "trial with two equally important endpoints"
drop_out_rate: "percentage of participants leaving study"
intention_to_treat: "analysis including all randomised subjects"
Kaplan_Meier_curve: "survival probability over time"
non_inferiority_trial: "test whether new treatment is not worse"
phase_I_trial: "first‑in‑human safety study"
phase_IV_trial: "post‑marketing surveillance study"
protocol_deviation: "departure from trial plan"
recruitment_rate: "speed of enrolling participants"
sporadic_case: "case occurring randomly rather than familial"
subgroup_analysis: "examining treatment effect in subsets"
wholesale_fda_approval: "full regulatory clearance for market"

#Epidemiology & public health
attack_rate: "proportion of exposed who become ill"
basic_reproduction_number: "average cases from one infected person"
case_fatality_rate: "deaths divided by diagnosed cases"
contact_tracing: "identifying and informing exposure contacts"
effective_reproduction_number: "real‑time transmission metric"
endemic: "constantly present disease within region"
herd_immunity_threshold: "proportion immune to stop spread"
incubation_period: "time from exposure to symptoms"
point_prevalence: "proportion with disease at a specific time"
serial_interval: "time between symptom onsets in transmission chain"
spillover_event: "pathogen jumps from animals to humans"
zoonosis: "disease transmitted from animals to humans"

#Metabolism & biochemistry
acetyl_CoA: "central metabolic cofactor carrying acetyl group"
allosteric_regulation: "enzyme activity modulated at site other than active site"
beta_oxidation: "fatty acid breakdown pathway"
carboxylase: "enzyme adding CO2 to substrate"
dehydrogenase: "enzyme removing hydrogen atoms"
enolase: "glycolytic enzyme converting 2‑phosphoglycerate"
gluconeogenesis: "formation of glucose from non‑carbohydrate sources"
glycogenolysis: "breakdown of glycogen"
hexokinase: "enzyme phosphorylating glucose"
isoenzyme: "different enzyme forms catalysing same reaction"
ketogenesis: "production of ketone bodies in liver"
lipogenesis: "fatty acid synthesis"
Michaelis_constant: "substrate concentration at half Vmax"
nicotinamide_adenine_dinucleotide: "redox cofactor (NAD⁺/NADH)"
oxidative_phosphorylation: "ATP synthesis using proton gradient"
prosthetic_group: "tightly bound non‑protein cofactor"
substrate_level_phosphorylation: "direct ATP formation in metabolic pathway"
urea_cycle: "conversion of ammonia to urea"

#Neuroscience
astrocyte: "glial cell supporting neurons"
brain_derived_neurotrophic_factor: "protein promoting neuron survival"
dendritic_spine: "small protrusion receiving synaptic input"
glutamatergic_synapse: "synapse using glutamate neurotransmitter"
ionotropic_receptor: "ligand‑gated ion channel"
long_term_potentiation: "persistent strengthening of synapses"
myelin_sheath: "insulating layer around axons"
NMDA_receptor: "glutamate receptor critical for plasticity"
parvalbumin_interneuron: "fast‑spiking inhibitory neuron subtype"
resting_membrane_potential: "baseline voltage across cell membrane"

#Miscellaneous
biobank: "repository storing biological samples"
chemotaxis: "cell movement toward chemical gradient"
exosome: "small extracellular vesicle for intercellular communication"
holoprotein: "protein without prosthetic group removed"
isoform: "alternative protein products from same gene"
metabolite: "small molecule in metabolism"
ontogeny: "development of an organism"
proteostasis: "maintenance of protein homeostasis"
quinone: "aromatic compound with two carbonyls"
transcriptome_assembly: "computational reconstruction of all transcripts"
biodegradation: "breakdown of substances by microorganisms"
biosimilar: "biologic drug highly similar to approved reference"
chemotaxis: "directed cell movement toward chemical gradient"
exosome: "extracellular vesicle carrying RNA/protein"
haptoglobin: "plasma protein binding free hemoglobin"
hydrophilicity: "affinity of molecule for water"
limulus_amebocyte_lysate: "assay detecting bacterial endotoxin"
mycotoxin: "toxic secondary metabolite of fungi"
photobleaching: "loss of fluorescence upon light exposure"
radionuclide: "radioactive isotope used in imaging/therapy"
siderophore: "iron‑chelating compound secreted by microbes"

# Immunology & microbiology
adjuvant_vaccine: "substance enhancing immune response to antigen"
apc: "antigen‑presenting cell that activates T cells"
cd4: "glycoprotein on helper T cells – HIV receptor"
cd8: "marker on cytotoxic T lymphocytes"
complement: "protein cascade that opsonises and lyses pathogens"
crm197: "non‑toxic diphtheria toxin used as conjugate carrier"
dendritic_cell: "potent antigen‑presenting immune cell"
elisa: "enzyme‑linked immunosorbent assay – protein quantification"
epitopes: "antigen portions recognized by antibodies or TCR"
fc_region: "antibody tail interacting with immune receptors"
helminth: "parasitic worm infecting humans or animals"
ifn_gamma: "interferon‑γ cytokine activating macrophages"
ige: "antibody class mediating allergic responses"
igm: "first antibody isotype produced in infection"
inflammasome: "cytosolic multiprotein complex activating IL‑1β"
innate_immunity: "non‑specific first line defence against pathogens"
lysozyme: "enzyme breaking bacterial cell walls"
macrophage: "phagocytic immune cell derived from monocytes"
monoclonal_antibody: "antibody from single B‑cell clone"
neutrophil: "abundant granulocyte that engulfs microbes"
pattern_recognition_receptor: "sensor detecting pathogen molecules"
plasmablast: "short‑lived antibody‑secreting B cell"
pma: "phorbol 12‑myristate 13‑acetate – T‑cell activator"
reverse_transcriptase: "enzyme converting RNA into DNA"
scfv: "single‑chain variable fragment – engineered antibody"
serotype: "variant of microorganism defined by antigenicity"
tcr: "T‑cell receptor recognizing peptide–MHC complexes"
tlr4: "Toll‑like receptor 4 – detects LPS"
trypsin: "protease that cleaves peptide bonds at lysine/arginine"
vaccinomics: "study of how genetics influences vaccine response"
variable_region: "antibody part that binds antigen"
wet_mount: "microscopy prep of fresh biological sample"

# Metabolism & biochemistry
acetyl‑coa: "central metabolite delivering acetyl groups"
adp_ribosylation: "post‑translational protein modification using ADP‑ribose"
aldolase: "glycolytic enzyme splitting fructose‑1,6‑bisphosphate"
allosteric_site: "regulatory site distinct from active site"
alpha_ketoglutarate: "TCA cycle intermediate, nitrogen acceptor"
aminotransferase: "enzyme transferring amino groups"
ampk: "AMP‑activated protein kinase – energy sensor"
beta_oxidation: "fatty‑acid degradation pathway in mitochondria"
bilirubin: "heme breakdown product causing jaundice"
calvin_cycle: "carbon fixation pathway in photosynthesis"
carboxylase: "enzyme adding CO₂ to substrate"
coenzyme_q10: "electron carrier in respiratory chain"
creatine_kinase: "enzyme buffering cellular ATP levels"
cytochrome_c: "mitochondrial electron carrier triggering apoptosis"
denaturation: "loss of protein secondary/tertiary structure"
exoenzyme: "enzyme secreted outside the cell"
flavin: "vitamin‑derived redox cofactor (FAD, FMN)"
glycogenolysis: "breakdown of glycogen to glucose"
glutathione: "tripeptide antioxidant detoxifying ROS"
hexokinase: "enzyme phosphorylating glucose to G6P"
krebs_cycle: "series generating NADH/FADH₂ from acetyl‑CoA"
lactate_dehydrogenase: "enzyme interconverting pyruvate and lactate"
mevalonate_pathway: "cholesterol and isoprenoid synthesis route"
nicotinamide: "vitamin B3 component of NAD⁺"
oxidative_phosphorylation: "ATP generation via electron transport"
peroxisome: "organelle performing fatty‑acid β‑oxidation and detox"
phosphofructokinase: "key glycolytic regulatory enzyme"
prosthetic_group: "tightly bound non‑protein cofactor"
shikimate_pathway: "aromatic amino‑acid biosynthesis route"
substrate_level_phosphorylation: "ATP synthesis coupled to metabolic reaction"
thioredoxin: "redox protein reducing disulfide bonds"
urea_cycle: "hepatic pathway detoxifying ammonia"

# Genetics & genomics
allele_specific_pcr: "PCR detecting single‑nucleotide variants"
antisense_oligonucleotide: "short DNA/RNA modulating gene expression"
array_cgh: "comparative genomic hybridisation for copy‑number"
base_editor: "CRISPR enzyme allowing single‑base conversion"
chromatin_immunoprecipitation: "assay mapping protein‑DNA interactions (ChIP)"
copy_number_variation: "genomic region present in abnormal copies"
crispr_cas9: "RNA‑guided nuclease for genome editing"
exome_sequencing: "sequencing only protein‑coding regions"
founder_mutation: "inherited variant from common ancestor"
fst: "fixation index measuring population differentiation"
gwas: "genome‑wide association study"
haplotype: "set of alleles inherited together"
indel: "insertion or deletion mutation"
linkage_disequilibrium: "non‑random association of alleles"
long_read_sequencing: "DNA sequencing producing >10 kb reads"
loop_extrusion: "cohesin‑mediated formation of chromatin loops"
microsatellite: "short tandem repeat of 1–6 bp"
mutagenesis: "experimental induction of mutations"
noncoding_rna: "RNA not translated into protein"
oligo_dT_primer: "primer binding poly‑A tail in RT‑PCR"
orf: "open reading frame potentially coding a protein"
polymerase_slippage: "error creating repeat length variation"
protospacer_adjacent_motif: "PAM – sequence required for Cas binding"
qtl: "quantitative trait locus"
radseq: "restriction site‑associated DNA sequencing"
replicon: "DNA molecule capable of replication"
splice_site: "junction between exon and intron"
structural_variant: "genomic rearrangement ≥50 bp"
synonymous_mutation: "nucleotide change not altering amino acid"
transversion: "purine↔pyrimidine substitution"
wave_front_sequencing: "fast single‑molecule nanopore protocol"

# Neuroscience
astrocyte: "glial cell maintaining neural environment"
axon_hillock: "neuronal region where action potentials start"
calcium_spike: "action potential in dendrites mediated by Ca²⁺"
dopamine: "neurotransmitter mediating reward"
endocannabinoid: "lipid neurotransmitter binding CB1/CB2"
glutamate_receptor: "ionotropic or metabotropic receptor for glutamate"
gliosis: "reactive change of glial cells after injury"
long_term_potentiation: "persistent strengthening of synapses"
microglia: "brain‑resident immune cells"
neuromuscular_junction: "synapse between motor neuron and muscle"
optic_chiasm: "crossing of optic nerves"
parkinsonism: "syndrome of rigidity and tremor"
perineuronal_net: "extracellular matrix surrounding neurons"
purinergic_signal: "cell signalling via ATP/adenosine"
resting_membrane_potential: "voltage across membrane at rest"
spike_frequency_adaptation: "decrease in neuron firing over time"
voltage_gated_channel: "ion channel opened by membrane voltage"

# Pharmacology & toxicology
affinity: "strength of drug‑target binding"
biotransformation: "metabolic conversion of drug to metabolite"
clearance: "volume of plasma cleared of drug per time"
competitive_inhibitor: "compound competing for active site"
dose_response_curve: "relation of dose to effect"
ed50: "dose achieving 50 % maximal effect"
first_pass_metabolism: "hepatic elimination before systemic circulation"
herg_blocker: "drug inhibiting cardiac hERG channel"
ligand_bias: "preferential signalling of receptor to subset pathways"
noael: "no‑observed‑adverse‑effect level in tox study"
pharmacogenomics: "study of genetic influence on drug response"
prodrug: "inactive compound converted to active drug in body"
qt_prolongation: "lengthening of cardiac QT interval"
therapeutic_window: "range between effective and toxic doses"
tmax: "time to reach peak concentration"
volume_of_distribution: "theoretical volume drug would occupy"

# Epidemiology & public health
attack_rate: "cumulative incidence in an outbreak"
case_fatality_ratio: "proportion of cases that die"
default_probability: "(placeholder) – remove if not biomedical"
doubled_time: "period required for cases to double"
effective_reproduction_number: "average secondary cases per infection (Rₑ)"
ev_percent: "vaccine efficacy percentage"
incubation_period: "time between infection and symptoms"
k_factor: "dispersion parameter for superspreading"
secondary_attack_rate: "transmission among household contacts"
serial_interval: "time between symptom onsets in chain"
transmission_bottleneck: "number of virions founding new infection"

# Bioinformatics tools & methods
blast: "Basic Local Alignment Search Tool"
docking: "computational prediction of ligand–protein binding"
edgeR: "Bioconductor package for RNA‑seq differential expression"
fastqc: "quality control tool for sequencing reads"
hisat2: "spliced aligner for RNA‑seq"
kallisto: "pseudo‑aligner for transcript quantification"
mafft: "multiple sequence alignment program"
prokka: "prokaryotic genome annotation pipeline"
rosetta: "suite for macromolecular modelling"
samtools: "utilities for SAM/BAM alignment files"
seqlogo: "graphical representation of motif conservation"
trimmomatic: "read trimming tool for Illumina data"
ucsc_genome_browser: "web platform for genomic data visualisation"

# Synthetic biology & biotechnology
biobrick: "standardised DNA part for synthetic circuits"
cell_free_expression: "protein synthesis in vitro without living cells"
crispr_interference: "dCas9 repression of transcription"
kill_switch: "engineered circuit causing cell death under conditions"
metabolic_engineering: "optimising pathways to overproduce compound"
optogenetics: "control of cell activity with light‑sensitive proteins"
phage_display: "technology displaying peptides on bacteriophages"
riboswitch: "RNA element regulating gene expression upon ligand"
toggle_switch: "bistable synthetic gene circuit"

# Endocrine & signaling molecules
aldosterone: "steroid hormone regulating salt balance"
angiotensin_II: "peptide hormone increasing blood pressure"
calcitonin: "hormone lowering blood calcium"
carboxypeptidase: "enzyme removing C‑terminal amino acids"
cholecystokinin: "gut hormone stimulating digestion"
cortisol: "glucocorticoid stress hormone"
erythropoietin: "kidney hormone inducing red blood cells"
follicle_stimulating_hormone: "pituitary hormone for gametogenesis"
GHRH: "growth‑hormone releasing hormone"
glucagon: "pancreatic hormone raising blood glucose"
growth_hormone: "pituitary hormone promoting growth"
insulin: "pancreatic hormone lowering blood glucose"
leptin: "adipose hormone suppressing appetite"
oxytocin: "peptide mediating uterine contraction & bonding"
parathyroid_hormone: "raises blood calcium levels"
prolactin: "pituitary hormone for lactation"
renin: "enzyme initiating RAAS pathway"
thyroid_stimulating_hormone: "pituitary hormone activating thyroid"
triiodothyronine: "active thyroid hormone T3"
vasopressin: "antidiuretic hormone concentrating urine"

# Cardiovascular biomarkers & factors
aortic_stiffness: "loss of vessel elasticity indicator"
brain_natriuretic_peptide: "heart failure biomarker BNP"
C_reactive_protein: "inflammatory plasma protein"
creatine_kinase_MB: "cardiac injury enzyme isoform"
d_dimer: "fibrin degradation product, clot marker"
fibrinogen: "soluble precursor of fibrin clot"
LDL_cholesterol: "low‑density lipoprotein – bad cholesterol"
HDL_cholesterol: "high‑density lipoprotein – good cholesterol"
myoglobin: "oxygen‑binding muscle protein, injury marker"
prothrombin_time: "clotting test measuring extrinsic pathway"
troponin_I: "cardiac‑specific injury biomarker"

# Neurotransmitters & receptors
acetylcholine: "neurotransmitter at neuromuscular junction"
alpha_synuclein: "protein aggregating in Parkinson disease"
dopamine_transport: "re‑uptake pump for dopamine"
GABA: "primary inhibitory neurotransmitter in CNS"
monoamine_oxidase: "enzyme degrading catecholamines"
NMDA_receptor: "glutamate‑gated calcium channel"
opioid_mu_receptor: "receptor mediating analgesia & euphoria"
serotonin: "monoamine mood neurotransmitter"

# Epigenetics & protein modifiers
DNA_methyltransferase: "enzyme adding methyl to cytosine"
histone_acetyltransferase: "adds acetyl to histone lysines"
histone_deacetylase: "removes acetyl groups from histones"
proteasome: "complex degrading ubiquitinated proteins"
ubiquitin_ligase: "E3 enzyme tagging proteins with ubiquitin"

# Innate immunity pattern sensors
cGAS_STING: "cytosolic DNA sensing pathway"
MAVS: "mitochondrial adaptor for RIG‑I antiviral signalling"
RIG_I: "RNA sensor detecting viral dsRNA"
TLR3: "Toll‑like receptor recognising dsRNA"
TLR9: "Toll‑like receptor recognising CpG DNA"
NLRP3_inflammasome: "multiprotein complex activating IL‑1β"

# Immuno‑checkpoint & T‑cell modulators
4_1BB: "costimulatory receptor on T cells (CD137)"
CAR_T_cell: "engineered T cell with chimeric antigen receptor"
CTLA4: "checkpoint receptor dampening T‑cell response"
LAG3: "inhibitory receptor on exhausted T cells"
OX40: "costimulatory molecule enhancing T‑cell survival"
PD1: "programmed cell death‑1 inhibitory receptor"
PDL1: "ligand for PD‑1 on tumors"
TIGIT: "inhibitory receptor on NK and T cells"
TIM3: "checkpoint receptor binding galectin‑9"

# RNA & gene‑editing tools
aptamer: "short nucleic acid binding specific targets"
base_editor_A_to_G: "adenine base editor converting A→G"
base_editor_C_to_T: "cytosine base editor converting C→T"
CRISPRa: "dCas9‑activator up‑regulating gene expression"
CRISPRi: "dCas9‑repressor down‑regulating gene expression"
prime_editor: "CRISPR nickase + RT for precise edits"
shRNA: "short hairpin RNA for gene knock‑down"
siRNA: "small interfering RNA silencing target mRNA"

# Viral & non‑viral delivery platforms
adeno_associated_virus: "small viral vector for gene therapy"
cell_penetrating_peptide: "short peptide ferrying cargo into cells"
electroporation: "electric pulses transiently permeabilise membranes"
lentiviral_vector: "integrating viral gene‑delivery system"
lipid_nanoparticle: "lipid vesicle delivering RNA/DNA drugs"
magnetofection: "magnetic nanoparticles enhancing transfection"
retroviral_vector: "virus integrating into dividing cells"
sonoporation: "ultrasound‑mediated membrane permeabilisation"

# Transposons & recombinases
cre_loxP: "site‑specific recombination system"
floxed_allele: "gene with loxP sites for conditional KO"
PhiC31_integrase: "phage enzyme inserting at att sites"
piggyBac_transposon: "cut‑and‑paste eukaryotic transposon"
Sleeping_Beauty: "synthetic Tc1/mariner transposase system"
Tol2_transposon: "fish transposon used in vertebrates"

# Multi‑omics & advanced sequencing
ATAC_seq: "assay for transposase‑accessible chromatin"
Bisulfite_seq: "DNA methylation sequencing method"
CUT_TAG: "cleavage under targets & tagmentation"
DNase_seq: "DNase I hypersensitivity mapping"
Hi_C: "genome‑wide chromatin contact mapping"
Nanopore_sequencing: "long‑read single‑molecule sequencing"
Ribo_seq: "ribosome profiling of translating mRNAs"
scRNA_seq: "single‑cell RNA sequencing"
spatial_transcriptomics: "RNA mapping with spatial resolution"

# Proteomics & metabolomics techniques
APEX_proximity_labeling: "peroxidase‑based mapping of protein neighbours"
BioID: "birA* proximity labelling technique"
phosphoproteomics: "large‑scale analysis of phosphorylated proteins"
SILAC: "stable isotope labelling for proteomics"
SWATH_MS: "data‑independent mass spectrometry acquisition"
TMT_labeling: "tandem mass tag multiplexed quantitation"
metabolomics_NMR: "metabolite profiling by NMR spectroscopy"
fluxomics: "measurement of metabolic fluxes"

# Imaging & microscopy innovations
expansion_microscopy: "physical tissue swelling for nanoscale imaging"
light_sheet_microscopy: "selective‑plane illumination imaging"
PALM_microscopy: "photoactivated localisation super‑resolution"
SIM_microscopy: "structured illumination microscopy"
STED_microscopy: "stimulated emission depletion imaging"
super_resolution_STED: "nanoscale depletion‑based imaging"
two_photon_microscopy: "deep tissue fluorescent imaging"
photoacoustic_tomography: "ultrasound generated by laser absorption"
array_tomography: "ultrathin section immunostaining 3D"
correlative_light_electron_microscopy: "combines LM & EM on same sample"

# Organoid & 3‑D culture technologies
bioprinting: "3‑D printing of living cells and biomaterials"
decellularised_matrix: "tissue scaffold devoid of cells"
gastruloid: "stem‑cell model of early embryo patterning"
induced_pluripotent_stem_cell: "reprogrammed adult stem cell"
lab_on_a_chip: "miniaturised microfluidic assay platform"
organ_on_chip: "microfluidic device modelling organ function"
organoid: "mini‑organ grown from stem cells"
spheroid: "3‑D aggregate of cultured cells"

# Advanced neuro & opto‑genetics
ChR2: "channelrhodopsin‑2 optogenetic activator"
DREADD: "designer receptor exclusively activated by designer drug"
halorhodopsin: "optogenetic chloride pump silencing neurons"
magnetogenetics: "magnetic control of cell activity"
optoacoustic_imaging: "light‑induced ultrasound for deep imaging"
thermogenetics: "temperature‑controlled neuronal modulation"

# Analytical & biophysical assays
biolayer_interferometry: "label‑free binding kinetics measurement"
cryo_FIB_milling: "focused ion beam thinning at cryo temp"
droplet_digital_PCR: "absolute quantification via droplet partitions"
electrochemiluminescence_immunoassay: "high‑sensitivity protein assay"
fluorescence_anisotropy: "measures molecular rotation/binding"
ISothermal_titration_calorimetry: "measures binding thermodynamics"
microscale_thermophoresis: "binding assay based on molecule movement"
nanoDSF: "intrinsic fluorescence thermal stability assay"
patch_clamp: "electrical recording from single cells"
SPR_surface_plasmon_resonance: "real‑time ligand binding sensor"

# Modern clinical‑trial designs
adaptive_trial: "design allowing protocol modifications mid‑study"
basket_trial: "same drug tested across multiple cancer types"
companion_diagnostic: "test identifying patients likely to benefit"
decentralized_trial: "study with remote participation"
synthetic_control_arm: "external real‑world data used as comparator"
umbrella_trial: "multiple drugs tested in a single cancer type"

# Public‑health analytics
real_world_evidence: "clinical data collected outside trials"
telemedicine_follow_up: "remote patient monitoring post‑therapy"
virtual_trial: "simulation modelling clinical outcomes"

# Miscellaneous cutting‑edge tools
bioorthogonal_click_chemistry: "in‑cell reaction tagging biomolecules"
copper_free_click: "azide–cyclooctyne reaction without Cu catalyst"
terahertz_imaging: "non‑ionising imaging using THz waves"
hyperspectral_imaging: "captures spectra per pixel for analysis"
lateral_flow_assay: "paper‑based diagnostic strip"
SHERLOCK_diagnostic: "CRISPR‑Cas13 nucleic acid detection"
DETECTR: "CRISPR‑Cas12 diagnostic platform"
PET_CT: "positron emission & X‑ray tomography combo"
diffusion_tensor_imaging: "MRI mapping white‑matter tracts"
functional_MRI: "blood‑oxygen‑level dependent brain imaging"

enhancer: "DNA element that boosts transcription of a nearby gene."
promoter: "DNA sequence where RNA polymerase binds to start transcription."
operon: "cluster of bacterial genes transcribed from one promoter as a single mRNA."
corepressor: "molecule that partners with a repressor protein to shut down transcription."
inducer: "small molecule that activates gene expression, often by disabling a repressor."
transcription_factor: "protein that binds DNA to regulate gene transcription."
heterochromatin: "tightly packed chromatin that is usually transcriptionally silent."
euchromatin: "loosely packed chromatin that is accessible for transcription."
cpg_island: "GC‑rich DNA stretch with many CG repeats, often found near gene promoters."
tata_box: "conserved promoter motif that helps position RNA polymerase II."
polyadenylation: "addition of a poly(A) tail to the 3′ end of an RNA."
spliceosome: "large RNA‑protein complex that removes introns from pre‑mRNA."
exon: "coding or untranslated RNA segment that remains after splicing."
intron: "non‑coding sequence removed from pre‑mRNA during splicing."
5'_utr: "untranslated region at the 5′ end of an mRNA."
3'_utr: "untranslated region at the 3′ end of an mRNA."
reverse_transcription: "synthesis of DNA from an RNA template by reverse transcriptase."
rrna: "ribosomal RNA—structural and catalytic RNA component of ribosomes."
trna: "transfer RNA that delivers amino acids to the ribosome."
snrna: "small nuclear RNA involved in splicing."
snorna: "small nucleolar RNA that guides rRNA chemical modifications."
mirna: "microRNA that suppresses gene expression post‑transcriptionally."
pirna: "PIWI‑interacting RNA that silences transposons in germ cells."
lncrna: "long non‑coding RNA (>200 nt) with diverse regulatory roles."
circrna: "covalently closed circular RNA molecule with regulatory functions."
ribozyme: "RNA molecule capable of catalyzing a chemical reaction."
rnase: "enzyme that degrades RNA."
rna_editing: "post‑transcriptional alteration of RNA nucleotide sequence."
rna_interference: "silencing of gene expression by small RNAs guiding RISC."
amino_acid: "building block of proteins containing an amino and carboxyl group."
polypeptide: "chain of amino acids linked by peptide bonds."
primary_structure: "linear sequence of amino acids in a protein."
secondary_structure: "local folding pattern such as α‑helix or β‑sheet."
alpha_helix: "right‑handed coiled secondary structure of proteins."
beta_sheet: "sheet‑like secondary structure formed by β‑strands."
tertiary_structure: "overall 3‑D shape of a single polypeptide chain."
quaternary_structure: "3‑D arrangement of multiple protein subunits."
chaperone: "protein that helps other proteins fold correctly."
ubiquitin: "small protein that tags substrates for degradation or other fates."
posttranslational_modification: "chemical change to a protein after translation."
glycosylation: "covalent attachment of sugar chains to proteins or lipids."
acetylation: "addition of an acetyl group to a molecule, often histones."
sumoylation: "covalent linkage of SUMO protein to a target, altering its function."
disulfide_bond: "covalent linkage between two cysteines stabilizing protein structure."
signal_transduction: "process by which a cell converts an external signal into a response."
phosphorylation_cascade: "series of sequential kinase activations transmitting a signal."
gtpase: "enzyme that hydrolyzes GTP, often acting as a molecular switch."
camp: "cyclic AMP—second messenger produced from ATP by adenylyl cyclase."
cgmp: "cyclic GMP—second messenger formed from GTP by guanylyl cyclase."
second_messenger: "small intracellular molecule that relays signals from receptors."
necroptosis: "programmed necrotic cell death triggered by specific signals."
ferroptosis: "iron‑dependent form of regulated cell death driven by lipid peroxides."
pyroptosis: "inflammatory programmed cell death involving gasdermin pores."
snp: "single‑nucleotide polymorphism—one‑base variation in DNA among individuals."
silent_mutation: "DNA change that does not alter the encoded amino acid."
missense_mutation: "mutation that substitutes one amino acid for another."
nonsense_mutation: "mutation converting a codon to a premature stop codon."
frameshift: "insertion or deletion that alters the mRNA reading frame."
insertion: "addition of nucleotide(s) into DNA."
deletion: "loss of nucleotide(s) from DNA."
duplication: "repetition of a DNA segment."
inversion: "reversal of a DNA segment within the chromosome."
translocation: "rearrangement in which DNA moves to a new chromosomal location."
recombination: "exchange of genetic material between DNA molecules."
crossing_over: "reciprocal exchange between homologous chromosomes during meiosis."
linkage: "tendency of genes close together to be inherited together."
pedigree: "family tree diagram showing inheritance of traits."
karyotype: "complete set of chromosomes visualized in a cell."
polyploidy: "condition of having more than two complete chromosome sets."
trisomy: "presence of an extra chromosome, giving three copies of one type."
monosomy: "loss of a chromosome; only one copy present."
mosaicism: "presence of genetically distinct cell populations in one individual."
genome: "entire genetic material of an organism."
transcriptome: "complete set of RNA transcripts in a cell or tissue."
proteome: "entire complement of proteins expressed in a cell or organism."
metabolome: "complete set of small‑molecule metabolites in a biological sample."
interactome: "full map of molecular interactions in a cell."
epigenome: "pattern of epigenetic marks across the genome."
microbiome: "community of microorganisms living in a particular environment."
qpcr: "quantitative PCR that measures DNA amplification in real time."
rt-pcr: "reverse‑transcription PCR to amplify cDNA from RNA."
ddpcr: "droplet digital PCR for absolute quantification of nucleic acids."
southern_blot: "method to detect specific DNA fragments by hybridization."
northern_blot: "method to detect specific RNA molecules by hybridization."
eastern_blot: "blotting method for detecting post‑translational modifications."
immunofluorescence: "fluorescent antibody labeling of proteins in cells or tissues."
flow_cytometry: "laser‑based counting and sorting of suspended cells."
facs: "fluorescence‑activated cell sorting—a flow cytometry‑based sorter."
confocal_microscopy: "optical microscopy giving high‑resolution sections via pinholes."
electron_microscopy: "imaging technique using electron beams for ultrastructural detail."
cryo-em: "electron microscopy of flash‑frozen samples, preserving native structure."
x-ray_crystallography: "method to determine atomic structures from protein crystals."
nmr_spectroscopy: "technique using nuclear magnetic resonance to study molecule structures."
mass_spectrometry: "analytical method that measures mass‑to‑charge of ionized molecules."
next_generation_sequencing: "high‑throughput DNA/RNA sequencing technologies."
illumina: "widely used sequencing platform based on reversible terminators."
shotgun_sequencing: "random fragmentation and parallel sequencing of DNA for assembly."
dna_microarray: "chip with probes to measure gene expression or variations simultaneously."
chip: "chromatin immunoprecipitation—captures DNA bound by a specific protein."
chip-seq: "ChIP combined with sequencing to map genome‑wide protein binding."
atac-seq: "assay for transposase‑accessible chromatin sequencing to map open DNA."
hi-c: "genome‑wide method to study 3‑D chromatin contacts."
rna-seq: "sequencing approach to quantify and discover RNA transcripts."
scrna-seq: "single‑cell RNA sequencing revealing cell‑by‑cell transcriptomes."
crispr-cas9: "editing system using Cas9 nuclease guided by RNA to cut DNA."
cas12: "class 2 CRISPR nuclease that can cleave single‑stranded DNA."
cas13: "CRISPR nuclease that targets RNA instead of DNA."
guide_rna: "RNA molecule that directs CRISPR nuclease to a target sequence."
pam_sequence: "protospacer adjacent motif—short DNA motif required for CRISPR targeting."
plasmid: "circular extrachromosomal DNA commonly used as a cloning vector."
vector: "vehicle DNA used to carry foreign genetic material into cells."
cloning: "making identical genetic copies of a DNA fragment or organism."
restriction_enzyme: "endonuclease cutting DNA at specific sequences."
taq_polymerase: "thermostable DNA polymerase from Thermus aquaticus used in PCR."
topoisomerase: "enzyme that relieves DNA supercoiling by cutting and rejoining strands."
dna_ligase: "ligase specific for sealing breaks in DNA."
sticky_ends: "single‑stranded DNA overhangs produced by staggered cuts."
blunt_ends: "DNA ends with no overhangs produced by straight cuts."
multiple_cloning_site: "engineered stretch of unique restriction sites in a vector."
expression_vector: "vector designed to produce protein from an inserted gene."
shuttle_vector: "vector that can replicate in two different host species."
bac: "bacterial artificial chromosome vector that carries large DNA inserts."
yac: "yeast artificial chromosome vector capable of carrying very large inserts."
cosmid: "plasmid‑phage hybrid vector for medium‑size DNA inserts."
transformation: "uptake of naked DNA by bacteria or yeast."
transfection: "introduction of nucleic acids into eukaryotic cells."
lipofection: "DNA or RNA delivery using lipid vesicles."
conjugation: "DNA transfer between bacteria through direct contact."
viral_vector: "virus engineered to deliver genetic material into cells."
lentivirus: "retrovirus vector capable of infecting dividing and non‑dividing cells."
adenovirus: "DNA virus often used as a delivery vector or vaccine platform."
retrovirus: "RNA virus that reverse‑transcribes its genome into host DNA."
transposon: "mobile genetic element that can move within the genome."
sleeping_beauty_transposon: "synthetic Tc1/mariner transposon system for gene delivery."
transgene: "foreign gene introduced into an organism."
fasta: "text‑based format for nucleotide or protein sequences and a search algorithm by that name."
alignment: "arrangement of sequences to identify regions of similarity."
multiple_sequence_alignment: "simultaneous alignment of three or more biological sequences."
phylogenetics: "study of evolutionary relationships among organisms or genes."
homology_modeling: "predicting protein structure based on homologous templates."
gene_ontology: "controlled vocabulary describing gene functions, processes, and locations."
nucleus: "membrane‑bound organelle containing the eukaryotic genome."
nucleolus: "nuclear substructure where ribosome assembly begins."
mitochondrion: "organelle that produces ATP via oxidative phosphorylation."
chloroplast: "plant organelle performing photosynthesis."
endoplasmic_reticulum: "membranous network for protein and lipid synthesis and folding."
plasma_membrane: "lipid bilayer enclosing the cell and controlling transport."
cytoskeleton: "network of actin filaments, microtubules, and intermediate filaments giving shape and facilitating movement."
microtubule: "hollow tubule built from tubulin dimers; part of the cytoskeleton."
intermediate_filament: "rope‑like cytoskeletal fiber providing mechanical strength."
centrosome: "microtubule‑organizing center of animal cells containing centrioles."
dna_double_helix: "two complementary polynucleotide strands wound around each other."
okazaki_fragment: "short DNA segment synthesized on the lagging strand during replication."
replication_fork: "Y‑shaped structure where DNA is being replicated."
kinetochore: "protein complex assembling on the centromere to connect chromosomes to spindle."
origin_of_replication: "DNA sequence where replication begins."
operator: "DNA segment in operons where repressors bind to regulate transcription."
tata_binding_protein: "core transcription factor that binds the TATA box."
helicase: "enzyme that unwinds DNA or RNA duplexes."
primase: "RNA polymerase that synthesizes primers for DNA replication."
dna_polymerase_iii: "main bacterial DNA replicative polymerase."
dna_polymerase_i: "bacterial polymerase that replaces RNA primers with DNA."
rnase_h: "enzyme that degrades RNA in RNA‑DNA hybrids."
chemokine: "cytokine that directs cell migration across chemotactic gradients."
growth_factor: "extracellular signaling protein that stimulates cell proliferation or differentiation."
interleukin: "class of cytokines primarily produced by leukocytes."
interferon: "cytokine family that induces antiviral responses."
hormone: "chemical messenger secreted into blood to regulate physiology."
receptor_tyrosine_kinase: "transmembrane receptor with cytoplasmic kinase activity activated by ligand binding."
gpcr: "G‑protein‑coupled receptor—seven‑pass membrane receptor activating heterotrimeric G proteins."
adaptor_protein: "protein that links signaling molecules without intrinsic enzymatic activity."
sh2_domain: "protein domain that binds phosphorylated tyrosines."
sh3_domain: "domain that binds proline‑rich motifs in proteins."
zinc_finger: "DNA‑binding motif stabilized by zinc ions."
leucine_zipper: "dimerization motif with leucines every seventh residue."
helix_turn_helix: "common DNA‑binding structural motif."
homeobox: "conserved 180‑bp DNA sequence encoding a developmental transcription factor domain."
wd40_repeat: "protein motif forming β‑propeller structures for protein–protein interactions."
phd_finger: "zinc‑binding protein domain recognizing histone modifications."
bromodomain: "protein module that binds acetylated lysines on histones."
chromodomain: "protein module recognizing methylated histone tails."
escherichia_coli: "model Gram‑negative bacterium widely used in biotechnology."
saccharomyces_cerevisiae: "brewer’s or baker’s yeast, a model eukaryote."
arabidopsis_thaliana: "small flowering plant serving as a model for plant biology."
drosophila_melanogaster: "fruit fly model organism in genetics."
caenorhabditis_elegans: "nematode worm used for developmental biology."
mus_musculus: "house mouse, principal mammalian model organism."
danio_rerio: "zebrafish model organism for vertebrate development."
glycolysis: "ten‑step pathway that converts glucose to pyruvate with ATP gain."
tca_cycle: "tricarboxylic acid cycle oxidizing acetyl‑CoA to CO₂ and harvesting energy."
electron_transport_chain: "series of respiratory complexes transferring electrons to oxygen."
pentose_phosphate_pathway: "metabolic pathway providing NADPH and ribose‑5‑phosphate."
fermentation: "anaerobic breakdown of organic molecules to generate ATP."
photosynthesis: "conversion of light energy into chemical energy in plants and algae."
g1_phase: "first gap phase between mitosis and DNA synthesis."
s_phase: "cell cycle phase during which DNA replication occurs."
g2_phase: "second gap phase following DNA synthesis, preceding mitosis."
m_phase: "mitotic phase encompassing nuclear and cell division."
g0_phase: "quiescent state where cells exit the cell cycle."
cyclin: "regulatory protein whose levels oscillate to control CDK activity."
cdk: "cyclin‑dependent kinase driving cell‑cycle transitions."
p53: "tumor‑suppressor transcription factor activated by DNA damage."
retinoblastoma_protein: "tumor suppressor that inhibits cell cycle until phosphorylated."
anaphase_promoting_complex: "ubiquitin ligase triggering sister‑chromatid separation."
spindle_assembly_checkpoint: "safeguard ensuring chromosomes are attached to spindle before anaphase."
mismatch_repair: "system correcting base‑pairing errors in DNA post‑replication."
nucleotide_excision_repair: "repair pathway removing bulky DNA lesions like thymine dimers."
base_excision_repair: "repair of small, non‑bulky base damage via base removal and resynthesis."
homologous_recombination: "error‑free DNA double‑strand break repair using a sister template."
nonhomologous_end_joining: "quick but error‑prone re‑ligation of DNA double‑strand breaks."
photolyase: "enzyme that reverses UV‑induced pyrimidine dimers using light energy."
dna_glycosylase: "enzyme that removes damaged bases initiating base‑excision repair."
dna_damage_response: "network of pathways sensing and repairing DNA lesions."
atm_kinase: "protein kinase activated by double‑strand breaks, triggering DDR."
atr_kinase: "kinase responding to replication stress and single‑strand DNA."
dna-pk: "kinase complex essential for nonhomologous end joining."
morphogen: "signaling molecule forming a gradient to pattern tissues during development."
gradient: "gradual change in concentration across space, guiding cellular processes."
induction: "process where one group of cells influences development of another."
cell_fate: "ultimate differentiated state a cell adopts."
totipotent: "capable of giving rise to all cell types including extraembryonic tissues."
pluripotent: "able to differentiate into almost any cell type of the body."
multipotent: "able to produce several related cell types."
stem_cell: "undifferentiated cell capable of self‑renewal and differentiation."
immunoglobulin: "antibody protein produced by B cells."
paratope: "antigen‑binding site on an antibody."
b_cell: "lymphocyte that produces antibodies."
t_cell: "lymphocyte mediating cellular immunity."
mhc: "major histocompatibility complex molecules presenting peptides to T cells."
mhc_class_i: "MHC molecules presenting intracellular peptides to CD8⁺ T cells."
mhc_class_ii: "MHC presenting extracellular peptides to CD4⁺ T cells."
epistasis: "interaction where one gene’s effect depends on another gene."
pleiotropy: "single gene influencing multiple phenotypic traits."
expressivity: "degree to which a genotype is expressed in an individual."
genetic_drift: "random changes in allele frequencies in small populations."
gene_flow: "movement of genes between populations via migration."
founder_effect: "reduced genetic diversity when a population is descended from a small number of founders."
bottleneck_effect: "genetic drift following a dramatic reduction in population size."
hardy-weinberg_equilibrium: "idealized state where allele frequencies remain constant across generations."
hydrogen_bond: "weak bond between a hydrogen atom and electronegative atom."
van_der_waals_interaction: "weak attraction due to transient dipoles between molecules."
ionic_bond: "electrostatic attraction between oppositely charged ions."
hydrophobic_interaction: "tendency of non‑polar molecules to avoid water and cluster together."
dna_methylation: "addition of methyl groups to DNA, often silencing genes."
histone_acetylation: "acetylation of histone lysines associated with active chromatin."
histone_methylation: "methyl modification of histone tails influencing chromatin state."
chromatin_remodeling: "ATP‑dependent repositioning of nucleosomes on DNA."
histone_tail: "N‑terminal region of histone proteins subject to modifications."
dna_demethylation: "removal of methyl groups from DNA, reactivating genes."
epigenetic_inheritance: "transmission of epigenetic marks through cell division or generations."
tumor_suppressor_gene: "gene that protects against cancer by controlling cell growth."
proto-oncogene: "normal version of a gene that can become an oncogene when mutated."
carcinogenesis: "multistep process of developing cancer."
telomerase: "reverse transcriptase that elongates telomeres."
gene_therapy: "treatment strategy that introduces genetic material to cure disease."
envelope: "lipid bilayer surrounding some viruses acquired from host membranes."
viral_genome: "nucleic‑acid genetic material of a virus (DNA or RNA)."
viral_replication: "process by which viruses produce new progeny inside a host cell."
lysogenic_cycle: "viral lifecycle where genome integrates and lies dormant in host DNA."
silencer: "DNA element that represses transcription when bound by specific proteins."
insulator: "DNA element that blocks interaction between enhancers and promoters."
mediator_complex: "multiprotein complex bridging transcription factors and RNA polymerase II."
elongation_factor: "protein that assists polymerases during chain elongation."
termination_factor: "protein that triggers release of RNA or ribosome to end synthesis."
ribosome: "large ribonucleoprotein complex that synthesizes proteins."
translation_initiation: "assembly of ribosome on mRNA and start codon selection."
translation_elongation: "sequential addition of amino acids to the growing polypeptide."
translation_termination: "release of completed polypeptide when a stop codon enters A site."
shine-dalgarno_sequence: "ribosome binding site in bacterial mRNAs upstream of start codon."
kozak_sequence: "optimal context around eukaryotic start codon enhancing translation."
initiation_factor: "protein aiding ribosome assembly at start codon."
release_factor: "protein that recognizes stop codons and promotes chain release."
peptidyl_transferase: "ribosomal rRNA catalytic center that forms peptide bonds."
aminoacyl-trna_synthetase: "enzyme linking a specific amino acid to its tRNA."
reading_frame: "division of mRNA sequence into consecutive, non‑overlapping codons."
codon: "triplet of nucleotides encoding an amino acid or stop signal."
anticodon: "tRNA triplet complementary to an mRNA codon."
wobble: "flexibility at the third codon position allowing one tRNA to pair with multiple codons."
start_codon: "AUG triplet signaling translation start."
stop_codon: "UAA, UAG, or UGA triplet that ends translation."
polycistronic: "mRNA encoding multiple proteins, common in bacteria."
monocistronic: "mRNA encoding a single protein, typical of eukaryotes."
nuclear_pore_complex: "large protein channel regulating transport across nuclear envelope."
importin: "transport receptor that carries proteins with NLS into the nucleus."
exportin: "transport receptor that moves cargo out of the nucleus."
signal_peptide: "short N‑terminal sequence directing a protein to the secretory pathway."
er_signal_sequence: "peptide tag targeting nascent proteins to the endoplasmic reticulum."
secretory_pathway: "route followed by proteins from ER through Golgi to exterior or organelles."
vesicle: "small membrane‑bound sac transporting materials in a cell."
endocytosis: "cellular uptake of material by membrane invagination."
exocytosis: "fusion of vesicles with the plasma membrane to release contents."
clip: "cross‑linking and immunoprecipitation technique to map RNA–protein interactions."
clip-seq: "high‑throughput sequencing version of CLIP."
ribo-seq: "ribosome profiling method sequencing ribosome‑protected mRNA fragments."
ribosome_profiling: "same as Ribo‑seq; maps translating ribosome positions on mRNAs."
gro-seq: "global run‑on sequencing measuring nascent RNA transcription."
pro-seq: "precision run‑on sequencing refining nascent transcription mapping."
dnase-seq: "sequencing of DNase I‑hypersensitive sites marking open chromatin."
mnase-seq: "sequencing of micrococcal nuclease‑digested chromatin to map nucleosomes."
bisulfite_sequencing: "method converting unmethylated cytosines to uracil to map DNA methylation."
medip: "methylated DNA immunoprecipitation used to enrich 5‑methylcytosine DNA."
faire: "formaldehyde‑assisted isolation of regulatory elements enriching open chromatin."
faire-seq: "sequencing‑based FAIRE for genome‑wide open chromatin mapping."
drop-seq: "single‑cell RNA‑seq method using droplets to isolate cells."
cite-seq: "single‑cell method combining RNA‑seq with surface protein barcoding."
atlas-seq: "genome‑wide mapping of chromatin contacts; advanced 3‑C derivative."
4c-seq: "circular chromosome conformation capture sequencing identifying contacts from one locus."
5c: "carbon copy chromosome conformation capture mapping many‑to‑many interactions."
base_editing: "genome editing converting one base to another without double‑strand breaks."
lim_domain: "zinc‑binding protein motif involved in protein interactions."
coiled_coil_domain: "structural motif of intertwined α‑helices for oligomerization."
ef-hand: "helix‑loop‑helix motif binding calcium ions."
ankyrin_repeat: "protein domain of tandem repeats mediating interactions."
heavy_chain: "larger polypeptide of an antibody or other multimeric protein."
light_chain: "smaller polypeptide paired with antibody heavy chain."
cdr: "complementarity‑determining region—variable loop in antibody binding site."
somatic_hypermutation: "high‑rate mutation process diversifying antibody genes."
class_switch_recombination: "B‑cell recombination changing antibody isotype."
microvillus: "microscopic membrane projection increasing absorptive surface."
cilium: "hair‑like projection involved in movement or sensing."
cell_wall: "rigid outer layer surrounding plant, fungal, or bacterial cells."
vacuole: "large membrane vesicle in plant or fungal cells for storage and osmotic balance."
embden–meyerhof_pathway: "classical glycolytic pathway breaking down glucose."
entner–doudoroff_pathway: "alternative glycolytic route in many bacteria."
glyoxylate_cycle: "modified TCA cycle enabling net glucose synthesis from acetyl‑CoA in plants and bacteria."
itraq: "isobaric tagging method for quantitative proteomics."
tmt: "tandem mass tag labeling for multiplexed proteomics."
two-dimensional_electrophoresis: "protein separation by isoelectric point and molecular weight."
housekeeping_gene: "gene expressed in all cells to maintain basic functions."
inducible_gene: "gene whose expression can be turned on by specific stimuli."
repressible_gene: "gene whose expression can be turned off by a regulator."
positive_regulation: "control mechanism that increases gene expression."
negative_regulation: "control mechanism that decreases gene expression."
vault_rna: "small RNA component of vault ribonucleoprotein particles."
tasirna: "trans‑acting siRNA guiding mRNA cleavage in plants."
tirna: "stress‑induced tRNA fragment involved in translational control."
quiescence: "reversible non‑dividing state of a cell."
dormancy: "period of metabolic inactivity in an organism or cell."
phagocytosis: "engulfment of large particles by a cell for degradation."
pinocytosis: "nonspecific uptake of extracellular fluid by small vesicles."
receptor-mediated_endocytosis: "selective uptake via ligand binding to surface receptors."
caveola: "flask‑shaped lipid raft invagination of the plasma membrane."
clathrin: "triskelion protein forming coated pits for endocytosis."
copii: "coat protein complex driving vesicle budding from ER to Golgi."
copi: "coat complex mediating retrograde Golgi‑to‑ER transport."
tight_junction: "cell junction sealing epithelial sheets and regulating paracellular transport."
adherens_junction: "cadherin‑based junction linking actin cytoskeletons of neighbors."
plasmodesma: "cytoplasmic channel connecting plant cells."
collagen: "major structural protein of connective tissues."
elastin: "elastic protein allowing tissues to stretch and recoil."
fibronectin: "glycoprotein linking cells to extracellular matrix."
laminin: "basement membrane glycoprotein promoting cell adhesion."
dynein: "minus‑end‑directed microtubule motor protein."
kinesin: "plus‑end‑directed microtubule motor protein."
myosin: "actin‑based motor protein generating force for muscle contraction and transport."
actomyosin: "contractile complex of actin filaments and myosin motors."
arp2/3_complex: "protein complex initiating branched actin polymerization."
formin: "protein promoting linear actin filament elongation."
phosphoinositide: "phosphorylated inositol lipid involved in signaling and membrane identity."
pip2: "phosphatidylinositol‑4,5‑bisphosphate, a plasma‑membrane phosphoinositide."
pip3: "phosphatidylinositol‑3,4,5‑trisphosphate, a signaling lipid activating Akt."
dag: "diacylglycerol—lipid second messenger activating PKC."
ip3: "inositol‑1,4,5‑trisphosphate—second messenger releasing Ca²⁺ from ER."
mapk: "mitogen‑activated protein kinase transmitting signals to the nucleus."
jak: "Janus kinase family of cytoplasmic tyrosine kinases in cytokine signaling."
stat: "signal transducer and activator of transcription proteins activated by JAKs."
nf-kb: "transcription factor central to immune and stress responses."
prion: "misfolded, self‑propagating protein causing transmissible neurodegeneration."
amyloid: "insoluble protein aggregate with β‑sheet structure found in diseases."
proteoglycan: "glycoprotein heavily modified with glycosaminoglycan chains."
glycosaminoglycan: "long, negatively charged polysaccharide in extracellular matrix."
heparan_sulfate: "sulfated glycosaminoglycan attached to proteoglycans."
microvesicle: "extracellular vesicle budding directly from the plasma membrane."
luciferase_assay: "reporter assay measuring gene expression via light emitted by luciferase."
reporter_gene: "detectable gene fused to a promoter to study regulation."
emsa: "electrophoretic mobility‑shift assay detecting DNA‑protein binding."
footprinting: "technique mapping protein‑DNA contacts by nuclease protection."
site-directed_mutagenesis: "method to create specific DNA sequence changes."
random_mutagenesis: "technique generating diverse mutations across a gene or genome."
rna_polymerase_i: "eukaryotic nuclear polymerase transcribing rRNA genes."
rna_polymerase_ii: "polymerase transcribing mRNA and many non‑coding RNAs."
rna_polymerase_iii: "polymerase transcribing tRNA and 5S rRNA genes."
immunoprecipitation: "capture of target proteins or complexes using specific antibodies."
co-immunoprecipitation: "immunoprecipitation used to test for protein–protein interactions."
agarose_gel_electrophoresis: "technique separating DNA or RNA fragments by size in an agarose matrix."
page: "polyacrylamide gel electrophoresis separating biomolecules by size/charge."
sds-page: "denaturing PAGE separating proteins primarily by size."
40s_subunit: "small subunit of eukaryotic ribosomes."
60s_subunit: "large subunit of eukaryotic ribosomes."
30s_subunit: "small subunit of bacterial ribosomes."
50s_subunit: "large subunit of bacterial ribosomes."
semiconservative_replication: "DNA replication where each daughter molecule retains one parental strand."
z-dna: "left‑handed DNA helix that forms under torsional stress."
a-dna: "right‑handed DNA form occurring in dehydrated conditions."
b-dna: "standard right‑handed DNA conformation in cells."
g-quadruplex: "four‑stranded DNA or RNA structure formed by guanine tetrads."
telomerase_rna_component: "RNA template subunit of telomerase."
telomerase_reverse_transcriptase: "catalytic protein subunit of telomerase."
endonuclease: "enzyme that cleaves internal phosphodiester bonds in nucleic acids."
exonuclease: "enzyme that removes nucleotides from the ends of DNA or RNA."
nickase: "endonuclease variant that cuts only one DNA strand."
alternative_splicing: "generation of multiple mRNAs from one gene by different exon usage."
sr_protein: "serine/arginine‑rich splicing factor family regulating exon selection."
hnrnp: "heterogeneous nuclear ribonucleoproteins binding pre‑mRNA and influencing processing."
branch_point: "adenosine within an intron that initiates splicing lariat formation."
adar: "adenosine deaminase acting on RNA performing A‑to‑I editing."
apobec: "cytidine deaminase family catalyzing C‑to‑U editing in RNA or DNA."
"""

# Pre-compiled global pattern for performance
_GLOBAL_PATTERN = None
_GLOBAL_GLOSSARY = None


class GlossaryManager:
    """
    Manages biomedical jargon simplification
    """

    def __init__(self, custom_glossary: Optional[Dict[str, str]] = None):
        """
        Initialize glossary manager

        Args:
            custom_glossary: Optional custom glossary to add/override
        """
        global _GLOBAL_GLOSSARY, _GLOBAL_PATTERN

        # Load base glossary - handle both uppercase and lowercase
        base_glossary = yaml.safe_load(GLOSSARY_YAML)

        # Normalize glossary keys to lowercase for consistent lookup
        self.glossary = {}
        for term, definition in base_glossary.items():
            self.glossary[term.lower()] = definition
            # Also store original case for special terms like PCR
            if term.isupper() or '-' in term:
                self.glossary[term] = definition

        # Add custom terms if provided
        if custom_glossary:
            for term, definition in custom_glossary.items():
                self.glossary[term.lower()] = definition
                if term.isupper() or '-' in term:
                    self.glossary[term] = definition

        # Use global pattern if available for performance
        if _GLOBAL_GLOSSARY and _GLOBAL_GLOSSARY == self.glossary:
            self.pattern = _GLOBAL_PATTERN
        else:
            # Create regex pattern for matching terms
            self._build_pattern()
            _GLOBAL_GLOSSARY = self.glossary.copy()
            _GLOBAL_PATTERN = self.pattern

        logger.info(f"Loaded glossary with {len(self.glossary)} terms")

    def _build_pattern(self):
        """Build regex pattern for matching glossary terms"""
        # Get unique terms (some may be duplicated due to case variations)
        unique_terms = set()
        for term in self.glossary.keys():
            # Add the term as-is
            unique_terms.add(term)
            # Add lowercase version
            unique_terms.add(term.lower())

        # Sort terms by length (longest first) to match longer phrases first
        sorted_terms = sorted(unique_terms, key=len, reverse=True)

        # Build pattern parts
        pattern_parts = []

        for term in sorted_terms:
            # Skip empty terms
            if not term:
                continue

            # Escape special regex characters
            escaped = re.escape(term)

            # Handle different term types
            if term.isupper() and len(term) > 1:
                # Uppercase abbreviations (PCR, DNA, etc.) - exact match only
                pattern_parts.append(f'\\b{escaped}\\b')
            elif '-' in term:
                # Hyphenated terms (wild-type, etc.) - match with word boundaries around whole term
                pattern_parts.append(f'\\b{escaped}\\b')
            elif ' ' in term:
                # Multi-word terms - match whole phrase
                pattern_parts.append(f'\\b{escaped}\\b')
            else:
                # Single words - case insensitive
                pattern_parts.append(f'\\b{escaped}\\b')
                
        greek_map = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ'}
        expanded = set()
        for term in unique_terms:
            for g_txt, g_sym in greek_map.items():
                if g_txt in term:
                    expanded.add(term.replace(g_txt, g_sym))
            # simple plural → singular (e.g., kinases → kinase)
            if term.endswith('s') and len(term) > 4:
                expanded.add(term[:-1])
        unique_terms.update(expanded)
        
        for term in expanded:
            escaped = re.escape(term)
            pattern_parts.append(f'\\b{escaped}\\b')

        # Create pattern
        if pattern_parts:
            pattern = '|'.join(pattern_parts)
            self.pattern = re.compile(pattern, re.IGNORECASE)
        else:
            # Empty pattern that matches nothing
            self.pattern = re.compile('(?!.*)')

    def add_terms(self, new_terms: Dict[str, str]):
        """
        Add new terms to glossary

        Args:
            new_terms: Dictionary of term -> definition
        """
        # Add to glossary with normalization
        for term, definition in new_terms.items():
            self.glossary[term.lower()] = definition
            if term.isupper() or '-' in term:
                self.glossary[term] = definition

        # Rebuild pattern
        self._build_pattern()

        # Clear global cache
        global _GLOBAL_GLOSSARY, _GLOBAL_PATTERN
        _GLOBAL_GLOSSARY = None
        _GLOBAL_PATTERN = None

        logger.info(f"Added {len(new_terms)} terms to glossary")

    def simplify_text(self, text: str, format: str = "html") -> str:
        """
        Add tooltips to jargon terms in text

        Args:
            text: Input text
            format: Output format ("html" or "markdown")

        Returns:
            Text with tooltips added
        """
        if format == "html":
            return self._simplify_html(text)
        elif format == "markdown":
            return self._simplify_markdown(text)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _simplify_html(self, text: str) -> str:
        """Add HTML tooltips to jargon terms"""
        # Track replacements to avoid overlapping
        replacements = []

        for match in self.pattern.finditer(text):
            term = match.group(0)
            start = match.start()
            end = match.end()

            # Look up definition
            term_lower = term.lower()
            definition = None

            # Try exact match first (for uppercase terms)
            if term in self.glossary:
                definition = self.glossary[term]
            elif term_lower in self.glossary:
                definition = self.glossary[term_lower]

            if definition:
                # Escape quotes in definition
                definition = definition.replace('"', '&quot;')
                replacement = f'<span class="jargon" title="{definition}">{term}</span>'
                replacements.append((start, end, replacement))

        # Apply replacements in reverse order to maintain positions
        result = text
        for start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]

        return result

    def _simplify_markdown(self, text: str) -> str:
        """Add markdown tooltips to jargon terms"""
        # First pass: collect all terms found in text
        found_terms = []
        term_positions = {}

        for match in self.pattern.finditer(text):
            term = match.group(0)
            term_lower = term.lower()

            # Look up definition
            definition = None
            if term in self.glossary:
                definition = self.glossary[term]
            elif term_lower in self.glossary:
                definition = self.glossary[term_lower]

            if definition and term_lower not in term_positions:
                found_terms.append((term, term_lower, definition))
                term_positions[term_lower] = []

            if term_lower in term_positions:
                term_positions[term_lower].append((match.start(), match.end()))

        # Second pass: replace terms with footnote markers
        replacements = []
        footnote_map = {}
        footnote_counter = 1

        for term, term_lower, definition in found_terms:
            if term_lower not in footnote_map:
                footnote_id = term_lower.replace(" ", "_").replace("-", "_")
                footnote_map[term_lower] = {
                    'id': footnote_id,
                    'number': footnote_counter,
                    'definition': definition
                }
                footnote_counter += 1

            # Mark all occurrences of this term
            for start, end in term_positions[term_lower]:
                footnote_ref = f'[^{footnote_map[term_lower]["id"]}]'
                replacements.append((start, end, text[start:end] + footnote_ref))

        # Apply replacements in reverse order
        result = text
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            result = result[:start] + replacement + result[end:]

        # Add footnotes at the end
        if footnote_map:
            footnotes = ["\n\n### Glossary\n"]
            for term_lower in sorted(footnote_map.keys()):
                info = footnote_map[term_lower]
                footnotes.append(f'[^{info["id"]}]: {info["definition"]}')

            result += '\n'.join(footnotes)

        return result

    def extract_jargon(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all jargon terms found in text

        Args:
            text: Input text

        Returns:
            List of (term, definition) tuples
        """
        found_terms = []
        seen = set()

        for match in self.pattern.finditer(text):
            term = match.group(0)
            term_lower = term.lower()

            # Skip if already found
            if term_lower in seen:
                continue

            # Look up definition
            definition = None
            if term in self.glossary:
                definition = self.glossary[term]
            elif term_lower in self.glossary:
                definition = self.glossary[term_lower]

            if definition:
                seen.add(term_lower)
                found_terms.append((term, definition))

        return found_terms

    def get_definition(self, term: str) -> Optional[str]:
        """
        Get definition for a specific term

        Args:
            term: Term to look up

        Returns:
            Definition or None if not found
        """
        # Try exact match first
        if term in self.glossary:
            return self.glossary[term]
        # Then try lowercase
        return self.glossary.get(term.lower())

    def search_terms(self, query: str) -> List[Tuple[str, str]]:
        """
        Search for terms containing query string

        Args:
            query: Search query

        Returns:
            List of (term, definition) tuples
        """
        query_lower = query.lower()
        results = []
        seen_definitions = set()

        for term, definition in self.glossary.items():
            # Skip duplicate definitions (from case variations)
            if definition in seen_definitions:
                continue

            if query_lower in term.lower() or query_lower in definition.lower():
                results.append((term, definition))
                seen_definitions.add(definition)

        # Sort by relevance
        def sort_key(item):
            term, definition = item
            term_lower = term.lower()

            # Exact match gets highest priority
            if term_lower == query_lower:
                return (0, len(term))
            # Term starts with query
            elif term_lower.startswith(query_lower):
                return (1, len(term))
            # Query in term
            elif query_lower in term_lower:
                return (2, len(term))
            # Query in definition
            else:
                return (3, len(term))

        results.sort(key=sort_key)

        return results

    @lru_cache(maxsize=128)
    def get_terms_for_category(self, category: str) -> List[Tuple[str, str]]:
        """
        Get all terms for a specific category

        Args:
            category: Category name

        Returns:
            List of (term, definition) tuples
        """
        category_keywords = {
            'molecular': ['protein', 'gene', 'DNA', 'RNA', 'enzyme', 'receptor', 'molecule'],
            'clinical': ['treatment', 'therapy', 'disease', 'cancer', 'patient', 'symptom'],
            'statistical': ['analysis', 'ratio', 'value', 'interval', 'trial', 'significance'],
            'technique': ['technique', 'method', 'assay', 'sequencing', 'microscopy', 'analysis'],
            'cell_biology': ['cell', 'cellular', 'mitosis', 'cycle', 'division', 'membrane'],
            'genetics': ['allele', 'mutation', 'chromosome', 'heredity', 'inheritance', 'gene']
        }

        if category not in category_keywords:
            return []

        keywords = category_keywords[category]
        results = []
        seen = set()

        for term, definition in self.glossary.items():
            if definition in seen:
                continue

            combined = term + " " + definition
            if any(kw in combined.lower() for kw in keywords):
                results.append((term, definition))
                seen.add(definition)

        return sorted(results)

    def export_glossary(self, format: str = "json") -> str:
        """
        Export glossary in different formats

        Args:
            format: Export format ("json", "yaml", "csv", "html")

        Returns:
            Formatted glossary string
        """
        # Remove duplicate entries (keep lowercase versions)
        unique_glossary = {}
        for term, definition in self.glossary.items():
            # Prefer the exact case version if it exists
            if term.lower() not in unique_glossary or term.isupper():
                unique_glossary[term] = definition

        if format == "json":
            return json.dumps(unique_glossary, indent=2, ensure_ascii=False)

        elif format == "yaml":
            return yaml.dump(unique_glossary, default_flow_style=False, allow_unicode=True)

        elif format == "csv":
            lines = ["term,definition"]
            for term, definition in sorted(unique_glossary.items()):
                # Escape quotes and commas
                definition = definition.replace('"', '""')
                if ',' in definition or '"' in definition:
                    definition = f'"{definition}"'
                lines.append(f"{term},{definition}")
            return "\n".join(lines)

        elif format == "html":
            html = "<dl>\n"
            for term, definition in sorted(unique_glossary.items()):
                html += f"  <dt><strong>{term}</strong></dt>\n"
                html += f"  <dd>{definition}</dd>\n"
            html += "</dl>"
            return html

        else:
            raise ValueError(f"Unknown format: {format}")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the glossary"""
        # Count unique definitions
        unique_definitions = len(set(self.glossary.values()))

        # Categorize terms
        categories = {
            'molecular': 0,
            'clinical': 0,
            'statistical': 0,
            'technique': 0,
            'cell_biology': 0,
            'genetics': 0,
            'other': 0
        }

        # Use cached category results
        for category in categories:
            if category != 'other':
                categories[category] = len(self.get_terms_for_category(category))

        # Count uncategorized
        categorized = sum(v for k, v in categories.items() if k != 'other')
        categories['other'] = unique_definitions - categorized

        return {
            'total_terms': len(self.glossary),
            'unique_definitions': unique_definitions,
            **categories
        }