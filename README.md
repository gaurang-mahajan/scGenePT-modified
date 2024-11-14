# scGenePT: Is language all you need for modeling single-cell perturbations?

## Model Description
scGenePT is a collection of single-cell models for perturbation prediction. It leverages the [scGPT](https://github.com/bowang-lab/scGPT)[1] foundation model for scRNAseq data by injecting language embeddings at the gene level into the model architecture. The language gene embeddings are obtained by embedding gene level information from different knowledge sources using LLMs. The knowledge sources used include NCBI gene descriptions, UniProt protein Summaries for protein coding genes - as inspired by the [genePT](https://github.com/yiqunchen/GenePT)[2] approach - and GO (Gene Ontology) Gene Molecular Annotations, across three different axes: Molecular Function, Biological Process and Cellular Component


## References
1. Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods (2024): 1-11. [Paper Link](https://www.nature.com/articles/s41592-024-02201-0) | [GitHub Repo](https://github.com/bowang-lab/scGPT) 
2. Chen, Yiqun, and James Zou. "GenePT: a simple but effective foundation model for genes and cells built from ChatGPT." bioRxiv (2024): 2023-10. [Paper Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824/) |  [GitHub Repo](https://github.com/yiqunchen/GenePT) 
5. Roohani, Yusuf, Kexin Huang, and Jure Leskovec. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." Nature Biotechnology 42.6 (2024): 927-935. [Paper Link](https://www.nature.com/articles/s41587-023-01905-6) | [GitHub Repo](https://github.com/snap-stanford/GEARS) 

