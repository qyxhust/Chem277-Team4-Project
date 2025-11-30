from utils.uniprot import *
import pandas as pd

def get_string_to_uniprot_mapping(string_ids):
    """
    Uses UniProt Web API to mappings from STRING to UniProt 
    """
    job_id = submit_id_mapping(
        from_db="STRING", to_db="UniProtKB", ids=string_ids
    )

    if check_id_mapping_results_ready(job_id):
        link = get_id_mapping_results_link(job_id)
        results = get_id_mapping_results_search(link)['results']

    string = [r['from'] for r in results]
    uniprot = [r['to']['primaryAccession'] for r in results]
    return {'STRING':string, 'UniProt':uniprot}


def get_ppi_data(path: str):
    """
    Gets PPI data in string format loaded in Pandas Dataframe
    """
    links = pd.read_csv(path, sep=' ')
    
    # Get STRING to UniProt id mapping
    unique_ids = pd.concat([links['protein1'], links['protein2']]).unique().tolist()
    mapping_df = get_string_to_uniprot_mapping(unique_ids)
    mapping = dict(zip(mapping_df['STRING'], mapping_df['UniProt']))
    
    # Convert links to UniProt
    protein1 = links['protein1'].map(mapping)
    protein2 = links['protein2'].map(mapping)
    links = pd.DataFrame({'protein1': protein1, 'protein2': protein2, 
                            'combined_score': links['combined_score']})
    return links

if __name__ == '__main__':
    links_path = 'data/raw/9606.protein.links.v12.0.txt'
    dataset_path = 'data/raw/41591_2025_3834_MOESM3_ESM.xlsx'
    save_path = 'data/cleaned'

    # Load and convert PPI data from STRING to UniProt ID
    links = get_ppi_data(links_path)
    dataset = pd.read_excel(dataset_path, 'SuppTbl5', header=1)

    # Drop duplicate proteins or rows with missing values
    dataset.drop_duplicates(subset='UniProt', inplace=True)
    dataset.dropna(how='any', inplace=True)
    unique_proteins = dataset['UniProt']

    # Remove all interactions for proteins not found in dataset
    links = links[links['protein1'].isin(unique_proteins) & \
                  links['protein2'].isin(unique_proteins)].reset_index(drop=True)
    
    links.to_csv(f'{save_path}/links.csv', index=False)
    dataset.to_csv(f'{save_path}/data.csv', index=False)
    