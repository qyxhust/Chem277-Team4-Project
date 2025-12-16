import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from uniprot_util import *

def load_gpnc_data(data_path):

    clincal_df = pd.read_csv(f'{data_path}/ClinicalV1_anonymized.csv')
    genetics_df = pd.read_csv(f'{data_path}/GeneticsV1_anonymized.csv')
    mass_spectrometry_df = pd.read_csv(f'{data_path}/MassSpecV1_anonymized.csv')
    person_mapping_df = pd.read_csv(f'{data_path}/PersonMappingV1_anonymized.csv')
    meta_df = pd.read_csv(f'{data_path}/SomalogicMetaV1_anonymized.csv')

    somalogic01V1_df = pd.read_csv(f'{data_path}/Somalogic01V1_anonymized.csv')
    somalogic02V1_df = pd.read_csv(f'{data_path}/Somalogic02V1_anonymized.csv')
    somalogic03V1_df = pd.read_csv(f'{data_path}/Somalogic03V1_anonymized.csv')
    somalogic04V1_df = pd.read_csv(f'{data_path}/Somalogic04V1_anonymized.csv')
    somalogic05V1_df = pd.read_csv(f'{data_path}/Somalogic05V1_anonymized.csv')
    somalogic06V1_df = pd.read_csv(f'{data_path}/Somalogic06V1_anonymized.csv')
    somalogic07V1_df = pd.read_csv(f'{data_path}/Somalogic07V1_anonymized.csv')
    somalogic08V1_df = pd.read_csv(f'{data_path}/Somalogic08V1_anonymized.csv')
    somalogic09V1_df = pd.read_csv(f'{data_path}/Somalogic09V1_anonymized.csv')
    somalogic10V1_df = pd.read_csv(f'{data_path}/Somalogic10V1_anonymized.csv')
    somalogic11V1_df = pd.read_csv(f'{data_path}/Somalogic11V1_anonymized.csv')
    somalogic12V1_df = pd.read_csv(f'{data_path}/Somalogic12V1_anonymized.csv')
    somalogic = [somalogic01V1_df, somalogic02V1_df, somalogic03V1_df, somalogic04V1_df, somalogic05V1_df, somalogic06V1_df, 
                somalogic07V1_df, somalogic08V1_df, somalogic09V1_df, somalogic10V1_df, somalogic11V1_df, somalogic12V1_df]
    somalogic_df = somalogic[0]
    for i in range(1, len(somalogic)):
        somalogic_df = somalogic_df.merge(somalogic[i].drop(['contributor_code', 'visit', 'sample_type'],axis=1), on='sample_id')

    return clincal_df, genetics_df, mass_spectrometry_df, person_mapping_df, somalogic_df, meta_df

def clean_clinical_df(clinical_df, meta_df):
    # Select baseline data and plasma samples 
    subset_meta_data = meta_df[meta_df['visit'] == 1]
    subset_meta_data = subset_meta_data[subset_meta_data['sample_matrix'].isin(['EDTA Plasma', 'Citrate Plasma'])]

    participants_x = clinical_df.merge(subset_meta_data, on='sample_id', how='right')
    participants_x.head(2)

    # Replace diagnosis with binary flags (1, 0)
    nd_cols = ['ad', 'pd', 'ftd', 'als']
    participants_x[nd_cols] = participants_x[nd_cols].replace([-1, 2], None)

    # Remove cases where both AD and sci_mci==1 + the AD participants have a CDR < 1 and 
    # those labeled as AD and another disease
    participants_x = participants_x[~((participants_x['ad']==1) & (participants_x['mci_sci']))]
    participants_x = participants_x[~((participants_x['ad']==1) & (participants_x['pd']))]
    participants_x = participants_x[~((participants_x['ad']==1) & (participants_x['ftd']))]
    participants_x = participants_x[~((participants_x['ad']==1) & (participants_x['als']))]
    participants_x = participants_x[~((participants_x['ad']==1) & (participants_x['cdr'] < 1.0))]

    # Remove sites where data is not 7K : Sites H, V and W (total of 381 participants)
    participants_x = participants_x[~(participants_x['contributor_code_x'].isin(['H', 'V', 'W']))]

    # Remove participants missing age or sex
    # Restrict dataset to those suitable for analyses (with computed diagnosis)
    participants_x = participants_x[~(participants_x['sex']==-1)]
    participants_x = participants_x[~(participants_x['age_at_visit']==-1)]

    # Drop all entries will missing values
    participants_x = participants_x.replace(-1, None)
    participants_x = participants_x[['sample_id'] + nd_cols].dropna()

    return participants_x

def clean_proteomic_data(somalogic_df, sd=5, nan_limit=8000):
    sequences = somalogic_df.filter(regex='seq_')

    # Replace -1 values with NaN and perform log2 transform
    sequences = sequences.replace(-1, np.nan).apply(pd.to_numeric, errors='coerce')
    sequences = np.log2(sequences)

    # Replace 5 SD outliers with NaN
    seq_mean = sequences.mean(skipna=True)
    seq_std = sequences.std(skipna=True)
    sequences = sequences.mask((sequences - seq_mean).abs() > sd * seq_std)

    # Remove any columns with >5000 NaNs
    nan_counts = sequences.isna().sum()
    sequences = sequences.drop(columns=nan_counts[nan_counts > nan_limit].index)

    sequences = pd.concat([somalogic_df[['sample_id']], sequences], axis=1)
    return sequences

if __name__ == '__main__':
    # Change data_path if data is stored elsewhere
    data_path = 'GNPCdata/GNPC Harmonised Dataset v1/Clinical'
    links_path = '9606.protein.links.v12.0.txt'
    save_path = '.'
    sd = 5
    nan_limit = 8000

    clinical_df, genetics_df, mass_spectrometry_df, person_mapping_df, somalogic_df, meta_df = load_gpnc_data(data_path)

    cleaned_clinical = clean_clinical_df(clinical_df, meta_df)
    cleaned_proteome = clean_proteomic_data(somalogic_df, sd=sd, nan_limit=nan_limit)

    cleaned_dataset = cleaned_clinical.merge(cleaned_proteome, on='sample_id', how='inner')
    cleaned_dataset.dropna(inplace=True)

    # Get protein mappings
    analyte_df = pd.read_csv(f'{data_path}/SomalogicAnalyteInfoV1_anonymized.csv')
    seq_mapping = analyte_df[['seq_id', 'uni_prot']].copy()
    seq_id = seq_mapping['seq_id'].str.split('-').apply(lambda x: f'seq_{x[0]}_{x[1]}')
    seq_mapping['seq_id'] = seq_id

    rename_dict = dict(zip(seq_mapping['seq_id'], seq_mapping['uni_prot']))
    cleaned_dataset.rename(columns=rename_dict, inplace=True)

    # Use UniProt API to retrieve STRING to UniProt ID and convert PPI links to UniProt
    links = pd.read_csv('9606.protein.links.v12.0.txt', sep=' ')
    string_ids = pd.concat([links['protein1'], links['protein2']]).unique().tolist()
    job_id = submit_id_mapping(
        from_db="STRING", to_db="UniProtKB", ids=string_ids
    )
    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)['results']

    strings = [r['from'] for r in results]
    uniprot = [r['to']['primaryAccession'] for r in results]
    string_to_uniprot = dict(zip(strings, uniprot))
    links['protein1'] = links['protein1'].map(string_to_uniprot)
    links['protein2'] = links['protein2'].map(string_to_uniprot)


    links_proteins = pd.concat([links['protein1'], links['protein2']], ignore_index=True)
    dataset_proteins = cleaned_dataset[cleaned_dataset.columns[6:]].columns
    unique_proteins = set(links_proteins).intersection(set(dataset_proteins))
    links = links[links['protein1'].isin(unique_proteins) & links['protein2'].isin(unique_proteins)].reset_index(drop=True)

    # Add edge node mapping
    protein_to_node = dict(zip(unique_proteins, range(len(unique_proteins))))
    links['src'] = links['protein1'].map(protein_to_node)
    links['dst'] = links['protein2'].map(protein_to_node)

    # Drop duplicate edges (i.e B-A to the A-B)
    pairs = pd.DataFrame(np.sort(links[['src', 'dst']], axis=1),
        index=links.index
    )
    links = links[~pairs.duplicated()]

    nodes = cleaned_dataset[list(unique_proteins)]

    # Generate graph data
    x = torch.tensor(nodes.values, dtype=torch.float)
    edge_index = torch.tensor(links[['src', 'dst']].values.T, dtype=torch.int)
    edge_weight = torch.tensor(links['combined_score'].values, dtype=torch.float).reshape(len(links['combined_score']), -1)

    nd_cols = ['ad', 'pd', 'ftd', 'als']
    labels = list(cleaned_dataset[nd_cols].sum(axis=1))
    y = torch.tensor(labels, dtype=torch.int)

    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

    torch.save(data, 'binary_label_data.pt')