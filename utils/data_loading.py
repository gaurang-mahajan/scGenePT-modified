from pathlib import Path
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import numpy as np
import pickle as pkl
from scgpt.utils import load_pretrained
import torch

# Dimension of GPT-3.5 ada embeddings
GPT_ADA_002_EMBED_DIM = 1536

# Mapping from embedding type to location on file of gene embeddings
GENE_EMBED_TYPE2LOCATION = {'ncbi_gpt' : 'genept/NCBI_gene_embedding_ada.pickle' ,
                           'ncbi+uniprot_gpt' : 'genept/NCBI+UniProt_embedding-gpt3.5-ada.pkl',
                           'go_c_gpt_concat': 'go_terms/GO_C_gene_embeddings_avg.pickle', 
                           'go_p_gpt_concat': 'go_terms/GO_P_gene_embeddings_avg.pickle', 
                           'go_f_gpt_concat': 'go_terms/GO_F_gene_embeddings_avg.pickle',
                           'go_all_gpt_concat': 'go_terms/GO_all_gene_embeddings_avg.pickle', 
                           'go_c_gpt_avg': 'go_terms/GO_C_gene_embeddings_avg.pickle', 
                           'go_p_gpt_avg': 'go_terms/GO_P_gene_embeddings_avg.pickle', 
                           'go_f_gpt_avg': 'go_terms/GO_F_gene_embeddings_avg.pickle',
                           'go_all_gpt_avg': 'go_terms/GO_all_gene_embeddings_avg.pickle'}

def get_embs_to_include(args):
    """
    Processes and returns a list containing the types of embeddings to include for gene representation based on model_type
    
    :param: command line args, containing model-type
    :returns list containing types of embeddings to include for gene representations
    """
    # GenePT Gene embeddings - either NCBI gene summaries or NCBI gene + UniProt protein summaries computed with GPT-3.5
    if args.model_type in ['genept_ncbi_gpt', 
                           'genept_ncbi+uniprot_gpt', 
                           'genept_ncbi+uniprot_gpt_no_attention', 
                           'genept_ncbi_gpt_no_attention']:
        embs_to_include = ['genePT_token_embs_gpt']
    # GO Gene Ontology Gene Annotations, computed with GPT-3.5
    # using averaging of gene annotations terms
    elif args.model_type in ['go_c_gpt_avg',  # Cellular Components
                             'go_f_gpt_avg',  # Moldecular Function
                             'go_p_gpt_avg',  # Biological Process
                             'go_all_gpt_avg']: # Average of GO_c + GO_f + GO_p
        embs_to_include = ['GO_token_embs_gpt_avg']
    # using concatenation of gene annotation terms
    elif args.model_type in ['go_c_gpt_concat', 
                             'go_f_gpt_concat',
                             'go_p_gpt_concat', 
                             'go_all_gpt_concat', 
                             'go_c_gpt_concat_no_attention', 
                             'go_f_gpt_concat_no_attention', 
                             'go_p_gpt_concat_no_attention', 
                             'go_all_gpt_concat_no_attention']:
        embs_to_include = ['GO_token_embs_gpt_concat']
    # scGPT token embeddings + scGPT counts embeddings + GenePT Gene embeddings - 
    # either NCBI gene summaries or NCBI gene + UniProt protein summaries computed with GPT-3.5
    elif args.model_type in ['scgenept_ncbi_gpt', 
                             'scgenept_ncbi+uniprot_gpt', 
                             'scgenept_ncbi_gpt_no_attention', 
                             'scgenept_ncbi+uniprot_gpt_no_attention']:
        embs_to_include = ['scGPT_counts_embs', 'scGPT_token_embs', 'genePT_token_embs_gpt']
    # scGPT token embeddings + scGPT counts embeddings + Gene Embeddings from GO Gene Ontology Annotations, computed with GPT-3.5
    # using averaging of gene annotations terms
    elif args.model_type in ['scgptgo_c_gpt_avg', 
                             'scgptgo_f_gpt_avg', 
                             'scgptgo_p_gpt_avg', 
                             'scgptgo_all_gpt_avg',]:
        embs_to_include = ['GO_token_embs_gpt_avg', 'scGPT_counts_embs', 'scGPT_token_embs']
    # using concatenation of gene annotation terms
    elif args.model_type in ['scgptgo_c_gpt_concat', 
                             'scgptgo_f_gpt_concat', 
                             'scgptgo_p_gpt_concat', 
                             'scgptgo_all_gpt_concat',]:
        embs_to_include = ['GO_token_embs_gpt_concat', 'scGPT_counts_embs', 'scGPT_token_embs']
    # scGPT token embeddings + scGPT counts embeddings + GenePT Gene embeddings + Gene Embeddings from GO Gene Ontology Annotations
    elif args.model_type in ['scgenept_ncbi+uniprot_gpt_go_c_gpt_concat', 
                             'scgenept_ncbi+uniprot_gpt_go_all_gpt_concat', 
                             'scgenept_ncbi+uniprot_gpt_go_f_gpt_concat', 
                             'scgenept_ncbi+uniprot_gpt_go_p_gpt_concat']:
        embs_to_include = ['scGPT_counts_embs', 'scGPT_token_embs', 'genePT_token_embs_gpt', 'GO_token_embs_gpt_concat']
    # scGPT counts embeddings + GenePT Gene embeddings 
    elif args.model_type in ['scgenept_ncbi_gpt_scgpt_counts', 
                             'scgenept_ncbi+uniprot_gpt_scgpt_counts']:
        embs_to_include = ['scGPT_counts_embs', 'genePT_token_embs']
    # scGPT counts embeddings + Gene Embeddings from GO Gene Ontology Annotations
    elif args.model_type in ['go_f_scgpt_counts', 
                             'go_p_scgpt_counts', 
                             'go_c_scgpt_counts', 
                             'go_all_scgpt_counts']:
        embs_to_include = ['scGPT_counts_embs', 'GO_token_embs']
    # scGPT token embeddings + scGPT counts embeddings
    elif args.model_type in ['scgpt']:
        embs_to_include = ['scGPT_counts_embs', 'scGPT_token_embs']
    # scGPT counts embeddings
    elif args.model_type in ['scgpt_counts']:
        embs_to_include = ['scGPT_counts_embs']
    # scGPT token embeddings
    elif args.model_type in ['scgpt_tokens']:
        embs_to_include = ['scGPT_tokens']
    return embs_to_include


def match_genes_to_scgpt_vocab(pretrained_model_location, pert_data, logger, special_tokens, dataset_name):
    """
    Parses a pre-trained scGPT model vocab and matches genes in a given pert_data corresponding to dataloaders from
    a dataset_name
    Code copied and modified from: https://scgpt.readthedocs.io/en/latest/tutorial_perturbation.html
    
    :param pretrained_model_location: location of scGPT pretrained model
    :param pert_data: PertData file containing training data
    :param logger: scGPT logger
    :param special_tokens: special tokens to add to the scGPT gene vocabulary; usually ["<pad", "<cls>", "<eoc>"]
    :param dataset_name: name of the dataset the pert_data object belongs to; likely one of ["norman", "adamson"]
    """
    model_dir = Path(pretrained_model_location)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    # initialize scGPT gene vocabulaty
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # check which genes are in the scGPT gene vocabulary
    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    if logger:
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
    
    # all genes in the dataset
    dataset_genes = pert_data.adata.var["gene_name"].tolist()
    vocab.set_default_index(vocab["<pad>"])
    
    # vocab gene_ids for the genes in the dataset, matched to the scGPT pretrained vocab
    dataset_gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in dataset_genes], dtype=int
    )
    
    n_genes = len(dataset_genes)
    
    # mapping from gene 2 gene_idx in the PertData adata file
    gene2idx = {}
    for i, g in enumerate(dataset_genes):
        gene2idx[g] = i
    return vocab, dataset_gene_ids, dataset_genes, gene2idx

def create_embs_w(genes, vocab, precomputed_embs_location, embed_dim, init_value = 0.1):
    """
    Creates an embedding matrix for a given list of genes, where each gene gets an embedding either from precomputed
    embeddings located at embedding_location, or by randomly initializing a vector with init_value.
    
    :param genes: genes to compute the embedding matrix for
    :param vocab: vocab mapping gene2index; needed to map correctly to the scGPT model architecture
    :param precomputed_embeddings_location: location of precomputed embeddings
    :param embed_dim: dimension of the precomputed embeddings, and consequently created embedding matrix 
    """
    with open(precomputed_embs_location, "rb") as fp:
        gene_embeddings = pkl.load(fp)
            
    embeds_m = np.random.uniform(-init_value, init_value, (len(vocab), embed_dim))
    found_genes = []
    found_genes_embeds_m = []
    count_missing = 0
    for i, gene in enumerate(genes):
        if gene in gene_embeddings:
            embed = gene_embeddings[gene]
            found_genes_embeds_m.append(embed)
            found_genes.append(gene)
        else:
            count_missing+=1
            
    print(f"Matched {len(genes) - count_missing} out of {len(genes)} genes in the GenePT-w embedding")
    gene_indices = vocab.lookup_indices(found_genes)
    embeds_m[gene_indices] = found_genes_embeds_m
    return embeds_m, found_genes

def initialize_genept_embeddings(embs_to_include, genes, vocab, model_type, pretrained_model_dir):
    """
    Initializes genept embeddings for a given set of genes, given that genePT embs should be included in the 
    list of gene representations.
    
    :param embs_to_include: list containing types of embeddings to include for gene representations
    :param genes: set of genes to map to genePT embeddings
    :param vocab: scGPT vocabulary
    :param model_type: model-type; determines the embeddings that get initialized
    """
    if 'genePT_token_embs_gpt' in embs_to_include or 'genePT_token_embs_llama' in embs_to_include:
        emb_info_type = model_type.split('_')[1] #'ncbi' or 'ncbi+uniprot'
        emb_type = model_type.split('_')[2] # 'gpt' or 'llama'
        
        print('Using', emb_info_type, 'genept embs, embedded with', emb_type)
        
        if emb_info_type == 'ncbi' and emb_type == 'gpt':
                emb_model_type = 'ncbi_gpt'   
        elif emb_info_type == 'ncbi+uniprot' and emb_type == 'gpt':
                emb_model_type = 'ncbi+uniprot_gpt'
                
        embed_dim = GPT_ADA_002_EMBED_DIM
        embeddings_location = pretrained_model_dir + GENE_EMBED_TYPE2LOCATION[emb_model_type]
        embeds, found_genes = create_embs_w(genes, vocab, embeddings_location, embed_dim)       
    else:
        embeds = []
        emb_info_type = None
        embed_dim = None
        found_genes = []
    return embeds, emb_info_type, embed_dim, found_genes



def initialize_go_embeddings(embs_to_include, genes, vocab, model_type, pretrained_model_dir):
    """
    Initializes GO (Gene Ontology) Annotations embeddings for a given set of genes, given that GO embs should be included in the 
    list of gene representations.
    
    :param embs_to_include: list containing types of embeddings to include for gene representations
    :param genes: set of genes to map to genePT embeddings
    :param vocab: scGPT vocabulary
    :param model_type: model-type; determines the embeddings that get initialized
    """     
    go_embs_to_include = {}
    found_genes_go = []
    
    if 'GO_token_embs_gpt_concat' in embs_to_include or 'GO_token_embs_gpt_avg' in embs_to_include:
        go_emb_type =  model_type.split('go_')[1].split('_')[0]
        print('Using', go_emb_type, 'GO embs')
        
        if 'GO_token_embs_gpt_avg' in embs_to_include:
            emb_model_type = f'go_{go_emb_type}_gpt_avg'
        elif 'GO_token_embs_gpt_concat' in embs_to_include:
            emb_model_type = f'go_{go_emb_type}_gpt_concat'
        
        embed_dim = GPT_ADA_002_EMBED_DIM
        embeddings_location = pretrained_model_dir + GENE_EMBED_TYPE2LOCATION[emb_model_type]
        embeds, found_genes = create_embs_w(genes, vocab, embeddings_location, embed_dim)
        go_embs_to_include[go_emb_type] = embeds
            
    else:
        go_embs_to_include = {}
        embed_dim = None
        found_genes = []
    return go_embs_to_include, embed_dim, found_genes

def load_pretrained_model(model, load_param_prefixs, verbose, model_file, device):
    """
    Loads the load_param_prefixs parameters from a pretrained model located in model_file into a given model instance. 
    Calls the scGPT load_pretrained function.
    
    :param model: model instance to load
    :param load_param_prefixs: list of parameter prefixes to load
    :param verbose: True if verbose
    :param mdoel_file: location of trained model
    :param device: device to load the model on
    """
    model = load_pretrained(model, torch.load(model_file, map_location=device), verbose=verbose, prefix=load_param_prefixs)
    return model




    