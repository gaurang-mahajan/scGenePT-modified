# scGenePT: Is language all you need for modeling single-cell perturbations?

## Model Description
scGenePT is a collection of single-cell models for perturbation prediction. It leverages the [scGPT](https://github.com/bowang-lab/scGPT) [1] foundation model for scRNAseq data by injecting language embeddings at the gene level into the model architecture. The language gene embeddings are obtained by embedding gene level information from different knowledge sources using LLMs. The knowledge sources used include NCBI gene descriptions, UniProt protein Summaries for protein coding genes - as inspired by the [genePT](https://github.com/yiqunchen/GenePT) [2] approach - and GO (Gene Ontology) Gene Molecular Annotations, across three different axes: Molecular Function, Biological Process and Cellular Component

![Example of gene representations for FOSB gene](FOSB_gene_example.png)

## Training 
### 1. Training <br>
scGenePT uses a pre-trained scGPT model and pre-computed gene embeddings. The files need to be under `models/pretrained/`, as described below. All of the pretrained models, as well as the assumed folder structure for model training can be found under this **[Google Drive Link](https://drive.google.com/drive/folders/1mit6pwRaykC28WQOSdPP-gFNbzKKJE2_?usp=drive_link)**.

**Download scGPT Pretrained Model** <br>
Pretrained model | Download link | Should be under
---- | --- | --- 
scGPT Model weights (whole-human) | [scGPT Google Drive Link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) <br> [CZI-Hosted Link](https://drive.google.com/drive/folders/1lNvQQpmHDSizEzbnKtBEheXT48OawqMc?usp=drive_link) | `models/pretrained/scgpt/` <br> - best_model.pt <br> - args.json <br> - vocab.json|


**Download Gene Embeddings** <br>
scGenePT can use multiple sources for textual gene annotations. The different sources and gene representations are described below. For GO Gene Annotations, only one embedding choice should be used.

model-type | Gene Embedding | Download link | Should be under 
---- | ---- | ---- | --- |
scgenept_ncbi| NCBI Gene summaries | [GenePT zenodo Link](https://zenodo.org/records/10833191) <br> [CZI-Hosted Link](https://drive.google.com/file/d/1wx-CFeqp5xFdynJrmUX6a73xxxfdZ0U4/view?usp=drive_link) | `models/pretrained/genept/` <br> NCBI_gene_embedding_ada.pickle
scgenept_ncbi+uniprot | NCBI Gene summaries + UniProt protein summaries| [CZI-Hosted Link](https://drive.google.com/file/d/1EyuQwY8B3DU3W2VBuiBSoJiJw7KHntu4/view?usp=drive_link)| `models/pretrained/genept` <br> NCBI+UniProt_embedding_ada.pkl
scgenept_go_c| GO Cellular Components Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1oGnxs56GqGQA5gaocg4uPf_UwtomJ0Tp/view?usp=drive_link)|`models/pretrained/genept` <br> GO_C_gene_embeddings_concat.pickle **or** GO_C_gene_embeddings_avg.pickle
scgenept_go_f| GO Molecular Function Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1ZGhHabXSg6eGkSCCMKf6o2-HvHIKpp3s/view?usp=drive_link) |`models/pretrained/genept` <br> GO_F_gene_embeddings_concat.pickle **or** GO_F_gene_embeddings_avg.pickle
scgenept_go_p| GO Biological Processes Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1pVRUpth4U8zhi1mRUgF5-lNMOg4jM9FF/view?usp=drive_link)| `models/pretrained/genept` <br> GO_P_gene_embeddings_concat.pickle **or** GO_P_gene_embeddings_avg.pickle
scgenept_go_all| Aggregation of GO-C + GO-F + GO-P| [CZI-Hosted Link](https://drive.google.com/file/d/1cQi6CtOEESXX9iVokwlcf_onVD3OmWNk/view?usp=drive_link)|  `models/pretrained/genept` <br> GO_all_gene_embeddings_concat.pickle **or** GO_all_gene_embeddings_avg.pickle

**Training Data** <br>

We use the processed versions of the Adamson and Norman datasets from [GEARS](https://github.com/snap-stanford/GEARS). Note that there are some differences in dataloaders between differrent versions, we trained and evaluated the models on GEARS v=0.0.2
To get the training, val and test splits:
```
from GEARS import PertData
dataset_name = 'norman' # or 'adamson'
pert_data = PertData("data/")
pert_data.load(data_name=dataset_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=val_batch_size)
```

**Training Script** <br>
`python train.py --model-type=scgenept_ncbi+uniprot --num-epochs=20 --dataset=norman --device=cuda:0 --rnd-seed=42`

The training arguments are the following: 

argument | definition | default value
--- | --- | --- 
**model-type** | type of model to train | scgenept_ncbi_gpt 
**num-epochs** | number of epochs to train the model for | 20 
**batch-size** | train batch_size | 64
**eval-batch-size** | validation batch_size | 64
**device** | device used for training | cude:0
**dataset** | dataset used for training | norman
**rnd-seed** | random seed used for training | 42
**max-seq-len** | number of genes to sample during training | 1536
**dropout** | dropout value to use during training | 0.2
**lr** | learning rate | 1e-4
**schedule-interval-lr** | schedule interval for lr | 1
**early-stop** | number of epochs to stop early if loss does not decrease | 10
**log-interval** | number of timesteps for which to log the loss | 100
**pretrained-model-dir** | directory of where the saved pretrained model directories are in | models/pretrained/
**outputs-dir** | directory where model outputs and metrics are saved | outputs/ 

## References
1. Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods (2024): 1-11. [Paper Link](https://www.nature.com/articles/s41592-024-02201-0) | [GitHub Repo](https://github.com/bowang-lab/scGPT) 
2. Chen, Yiqun, and James Zou. "GenePT: a simple but effective foundation model for genes and cells built from ChatGPT." bioRxiv (2024): 2023-10. [Paper Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824/) |  [GitHub Repo](https://github.com/yiqunchen/GenePT) 
5. Roohani, Yusuf, Kexin Huang, and Jure Leskovec. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." Nature Biotechnology 42.6 (2024): 927-935. [Paper Link](https://www.nature.com/articles/s41587-023-01905-6) | [GitHub Repo](https://github.com/snap-stanford/GEARS) 

