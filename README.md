# scGenePT: Is language all you need for modeling single-cell perturbations?

## Model Description
scGenePT is a collection of single-cell models for perturbation prediction. It leverages the [scGPT](https://github.com/bowang-lab/scGPT) [1] foundation model for scRNAseq data by injecting language embeddings at the gene level into the model architecture. The language gene embeddings are obtained by embedding gene level information from different knowledge sources using LLMs. The knowledge sources used include NCBI gene descriptions, UniProt protein Summaries for protein coding genes - as inspired by the [genePT](https://github.com/yiqunchen/GenePT) [2] approach - and GO (Gene Ontology) Gene Molecular Annotations, across three different axes: Molecular Function, Biological Process and Cellular Component

![Example of gene representations for FOSB gene](FOSB_gene_example.png)

## Training 

### Step 1: Download pretrained models and pre-computed gene Embeddings <br>
scGenePT uses a pre-trained scGPT model and pre-computed gene embeddings. The files need to be under `models/pretrained/`, as described below. All of the trained models, as well as the assumed folder structure for model training can be found under this **[Google Drive Link](https://drive.google.com/drive/folders/12h1hL3cJF3W0VG92JqGJ1-4R-2nDXbzc?usp=drive_link)**.

**Download scGPT Pretrained Model** <br>
Pretrained model | Download link | Should be under
---- | --- | --- 
scGPT Model weights (whole-human) | [scGPT Google Drive Link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) <br> [CZI-Hosted Link](https://drive.google.com/drive/folders/1lNvQQpmHDSizEzbnKtBEheXT48OawqMc?usp=drive_link) | `models/pretrained/scgpt/` <br> - best_model.pt <br> - args.json <br> - vocab.json|


**Download Gene Embeddings** <br>
scGenePT can use multiple sources for textual gene annotations. The different sources and gene representations are described below, together with the download links. If you're only interested in using one type of gene embeddings, you only need to download those embeddings only. 

**All gene embeddings can be found under [this link](https://drive.google.com/drive/folders/191d8uXaoUNvvZ8DZHzlR1O6BK9vLnqqy?usp=drive_link).**

Gene Embedding | Download link | Should be under 
---- | ---- | --- |
NCBI Gene summaries | [GenePT zenodo Link](https://zenodo.org/records/10833191) <br> [CZI-Hosted Link](https://drive.google.com/file/d/1wx-CFeqp5xFdynJrmUX6a73xxxfdZ0U4/view?usp=drive_link) | `models/gene_embeddings/` <br> NCBI_gene_embeddings-gpt3.5-ada.pickle
NCBI Gene summaries + UniProt protein summaries| [CZI-Hosted Link](https://drive.google.com/file/d/1EyuQwY8B3DU3W2VBuiBSoJiJw7KHntu4/view?usp=drive_link)| `models/gene_embeddings/` <br> NCBI+UniProt_embeddings-gpt3.5-ada.pkl
GO Cellular Components Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1oGnxs56GqGQA5gaocg4uPf_UwtomJ0Tp/view?usp=drive_link)|`models/gene_embeddings/` <br> GO_C_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_C_gene_embeddings-gpt3.5-ada_avg.pickle
GO Molecular Function Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1ZGhHabXSg6eGkSCCMKf6o2-HvHIKpp3s/view?usp=drive_link) |`models/gene_embeddings/` <br> GO_F_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_F_gene_embeddings-gpt3.5-ada_avg.pickle
GO Biological Processes Annotations| [CZI-Hosted Link](https://drive.google.com/file/d/1pVRUpth4U8zhi1mRUgF5-lNMOg4jM9FF/view?usp=drive_link)| `models/gene_embeddings/` <br> GO_P_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_P_gene_embeddings-gpt3.5-ada_avg.pickle
Aggregation of GO-C + GO-F + GO-P| [CZI-Hosted Link](https://drive.google.com/file/d/1cQi6CtOEESXX9iVokwlcf_onVD3OmWNk/view?usp=drive_link)|  `models/gene_embeddings/` <br> GO_all_gene_embeddings-gpt3.5-ada_concat.pickle **or** GO_all_gene_embeddings-gpt3.5-ada_avg.pickle

### Step 2: Environment setup

We highly recommend creating a virtual environment. Model output has not been tested outside of the packages below.
```
conda create -y --name scgenept python=3.10
source activate scgenept
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install scgpt "flash-attn<1.0.5"
```

### Step 3: Train a model <br>

**Training Data** <br>

We use the processed versions of the Adamson and Norman datasets from [GEARS](https://github.com/snap-stanford/GEARS). Note that there are some differences in dataloaders between differrent versions, we trained and evaluated the models on GEARS v=0.0.2. This code snippet is already embedded in the the codebase, so no additional work is needed to train on these datasets. To train on new datasets, data would have to be processed in a PertData object in order to be compatible with the current code.

```python
from GEARS import PertData
dataset_name = 'norman' # or 'adamson'
pert_data = PertData("data/")
pert_data.load(data_name=dataset_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=val_batch_size)
```

**Training Script** <br>
Once the pre-trained scGPT model and pre-computed gene embeddings have been downloaded under `models/pretrained/`, as described above, scGenePT can be trained using: 

`python train.py --model-type=scgenept_ncbi+uniprot_gpt --num-epochs=20 --dataset=norman --device=cuda:0 --rnd-seed=42`

The training arguments are the following: 

argument | definition | default value
--- | --- | --- 
**model-type** | type of model to train; see info in data_loading.py for all model variations| scgenept_ncbi_gpt 
**num-epochs** | number of epochs to train the model for | 20 
**batch-size** | train batch_size | 64
**eval-batch-size** | validation batch_size | 64
**device** | device used for training | cuda:0
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

For **model_type** the following options are possible:
**scGPT + NCBI Gene Card/NCBI Gene Card + UniProt Protein Summaries**
- scgenept_ncbi_gpt
- scgenept_ncbi+uniprot_gpt

**scGPT + GO Gene Annotations**  
- scgenept_go_f_gpt
- scgenept_go_c_gpt
- scgenept_go_p_gpt
- scgenept_go_all_gpt

**Gene Annotations**
- genept_ncbi_gpt
- genept_ncbi+uniprot_gpt
- go_f_gpt_concat
- go_c_gpt_concat
- go_p_gpt_concat
- go_all_gpt_concat

**scGPT + NCBI Gene Card/NCBI Gene Card + UniProt Protein Summaries + GO Gene Annotations**  
- scgenept_ncbi+uniprot_gpt_go_c_gpt_concat
- scgenept_ncbi+uniprot_gpt_go_f_gpt_concat
- scgenept_ncbi+uniprot_gpt_go_p_gpt_concat
- scgenept_ncbi+uniprot_gpt_go_all_gpt_concat
- scgenept_ncbi+uniprot_gpt_go_c_gpt_concat

**scGPT**
- scgpt
- scgpt_counts
- scgpt_tokens

For each of the model types, a suffix **_no_attention** can be added, which means that the model won't use scGPT pre-trained attention.

## Inference

**Tutorials**

- [scgenept_tutorial Google Colab](https://colab.research.google.com/drive/12Lg_dNy55-ii69hsfc3_bLJeVS1eNsDB) - Tutorial showcasing how to use trained scGenePT models in inference mode for perturbation prediction. It uses models fine-tuned on the Norman dataset and offers examples of predicting post-perturbation expression responses for single and two-gene perturbations.

**Step 1**: The following files need to be downloaded beforehand:
- **Download scGPT Pretrained Model** - the scGPT model needs to be under `models/pretrained/scgpt` 
- **Download Gene Embeddings** - the gene embeddings files should be under `models/gene_embeddings`. 
- **Download Trained scGenePT models from the scGenePT Model Zoo** - they should be under `models/finetuned`

**Step 2**: The environment needs to be setup properly:
```
conda create -y --name scgenept python=3.10
source activate scgenept
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install scgpt "flash-attn<1.0.5"
```

## scGenePT Model Zoo
Trained scGenePT Models can be downloaded from this Google Drive [link](https://drive.google.com/drive/folders/1U9PodoV7A-Dkk-GemmLB_AzmkgE-owp_?usp=drive_link)

Model | Description | Download link 
--- | --- | ---
scgenept_ncbi | scGPT + NCBI Gene Card Summaries | [link](https://drive.google.com/drive/folders/1AKLLWylPplFMZiaPiNJJejiB5Vt6gv9d?usp=drive_link) 
scgenept_ncbi+uniprot | scGPT + NCBI Gene Card Summaries + UniProt Protein Summaries | [link](https://drive.google.com/drive/folders/1r5awtb1rrYUlRzKwoDU7fsF8jQ_cLx6O?usp=drive_link)
scgenept_go_c | scGPT + GO Cellular Components Annotations | [link](https://drive.google.com/drive/folders/18e0RpI3umdAiQyqcSZOHRFLq-ajuTX-O?usp=drive_link)
scgenept_go_f | scGPT + GO Molecular Functions Annotations | [link](https://drive.google.com/drive/folders/1ewuCsKPHjx0Dyek3lex3LosLl6dGYy7B?usp=drive_link)
scgenept_go_p | scGPT + GO Biological Processes Annotations | [link](https://drive.google.com/drive/folders/1vdgOlZ3HzwqEf_s6mJN699yoqyOhNGwB?usp=drive_link)
scgenept_go_all | scGPT + GO_F + GO_C + GO_P | [link](https://drive.google.com/drive/folders/1BNbNX-1KZE4BbIfiRyExWtWTgI-lQHDX?usp=drive_link)
scgpt | scGPT | [link](https://drive.google.com/drive/folders/1rGtTDG7l5bbfIxLbSVr_mbQ0uZbXpTMe?usp=drive_link)

## Cite Us
If you use scGenePT in your analyses, please cite us:

**Paper**: Istrate, Ana-Maria, Donghui Li, and Theofanis Karaletsos. "scGenePT: Is language all you need for modeling single-cell perturbations?." bioRxiv (2024): 2024-10. [bioRxiv Link](https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1)

```
@article{istrate2024scgenept,
  title={scGenePT: Is language all you need for modeling single-cell perturbations?},
  author={Istrate, Ana-Maria and Li, Donghui and Karaletsos, Theofanis},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## References
1. Cui, Haotian, et al. "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." Nature Methods (2024): 1-11. [Paper Link](https://www.nature.com/articles/s41592-024-02201-0) | [GitHub Repo](https://github.com/bowang-lab/scGPT) 
2. Chen, Yiqun, and James Zou. "GenePT: a simple but effective foundation model for genes and cells built from ChatGPT." bioRxiv (2024): 2023-10. [Paper Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824/) |  [GitHub Repo](https://github.com/yiqunchen/GenePT) 
5. Roohani, Yusuf, Kexin Huang, and Jure Leskovec. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." Nature Biotechnology 42.6 (2024): 927-935. [Paper Link](https://www.nature.com/articles/s41587-023-01905-6) | [GitHub Repo](https://github.com/snap-stanford/GEARS) 

