import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import os
import sys
import glob
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def find_latest_cluster_file():
    """Finds the most recent cluster assignment CSV in data/04-predictions."""
    base_dir = os.path.join(project_root, 'data', '04-predictions')
    # Find all run directories
    run_dirs = sorted(glob.glob(os.path.join(base_dir, 'run_*')), reverse=True)
    
    if not run_dirs:
        print("No prediction runs found!")
        return None, None
    
    latest_dir = run_dirs[0]
    print(f"Checking latest run directory: {latest_dir}")
    
    # Find csv files in the latest run dir
    csv_files = glob.glob(os.path.join(latest_dir, 'protein_clusters_*.csv'))
    
    if not csv_files:
        print("No cluster CSV files found in the latest run directory.")
        return None, None
        
    # Pick the one with highest silhouette score if multiple exist, or just the first one
    # Analyze.py saves them as 'protein_clusters_...silX.XX.csv', let's pick the one with highest sil if possible
    # Or just simply pick the last one in the list which usually corresponds to the best config saved last
    target_file = csv_files[-1] 
    print(f"Found target cluster file: {target_file}")
    return target_file, latest_dir

def run_enrichment_for_cluster(gene_list, description, out_dir):
    """Runs Enrichr for a given list of genes."""
    if len(gene_list) < 5:
        print(f"  [Skip] Too few genes ({len(gene_list)}) for enrichment.")
        return
    
    print(f"  Running enrichment for {description} ({len(gene_list)} genes)...")
    
    try:
        # Database selection: GO Biological Process and KEGG
        # You can add 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023' if needed
        gene_sets = ['GO_Biological_Process_2023', 'KEGG_2021_Human']
        
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism='Human',
            outdir=None, # Don't write default files, we will handle output
            cutoff=0.05  # Adjusted P-value cutoff
        )
        
        results = enr.results
        
        # Filter for significant results
        sig_results = results[results['Adjusted P-value'] < 0.05].copy()
        
        if sig_results.empty:
            print("    No significant terms found.")
            return

        # Save full results
        clean_desc = description.replace(' ', '_').replace('/', '-')
        csv_name = f"enrichment_{clean_desc}.csv"
        sig_results.to_csv(os.path.join(out_dir, csv_name))
        
        # Plotting Top 10 Terms for each Gene Set
        for gs in gene_sets:
            subset = sig_results[sig_results['Gene_set'] == gs].head(10)
            if subset.empty:
                continue
                
            # Calculate -log10(P-value) for better visualization
            subset['log_prob'] = -np.log10(subset['Adjusted P-value'])
            subset = subset.sort_values('log_prob', ascending=True) # Sort for bar plot
            
            plt.figure(figsize=(10, 6))
            plt.barh(subset['Term'], subset['log_prob'], color='steelblue')
            plt.xlabel('-log10(Adjusted P-value)')
            plt.title(f'Top 10 {gs} - {description}')
            plt.tight_layout()
            
            plot_name = f"plot_{clean_desc}_{gs}.png"
            plt.savefig(os.path.join(out_dir, plot_name), dpi=300)
            plt.close()
            
        print(f"    Saved results and plots to {out_dir}")
        
    except Exception as e:
        print(f"    Error running enrichment: {e}")
        print("    (Note: This requires internet access to connect to Enrichr)")

def main():
    print("🚀 Starting Ontology & Pathway Analysis...")
    
    # 1. Find Data
    cluster_file, run_dir = find_latest_cluster_file()
    if not cluster_file:
        return

    # Create output directory for ontology
    ontology_dir = os.path.join(run_dir, 'ontology_analysis')
    os.makedirs(ontology_dir, exist_ok=True)
    
    # 2. Load Clusters
    df = pd.read_csv(cluster_file)
    
    # Check columns
    if 'GeneSymbol' not in df.columns or 'Cluster' not in df.columns:
        print("Error: CSV must contain 'GeneSymbol' and 'Cluster' columns.")
        return

    # 3. Iterate Clusters
    clusters = sorted(df['Cluster'].unique())
    
    print(f"Found {len(clusters)} groups (including noise if present).")

    for cluster_id in clusters:
        cluster_name = str(cluster_id)
        if 'Noise' in cluster_name or cluster_name == '-1':
            print(f"\nSkipping Noise cluster...")
            continue
            
        print(f"\nProcessing {cluster_name}...")
        
        # Get genes for this cluster
        genes = df[df['Cluster'] == cluster_id]['GeneSymbol'].tolist()
        # Clean gene names if necessary (remove whitespace)
        genes = [str(g).strip() for g in genes]
        
        # Run Analysis
        run_enrichment_for_cluster(genes, f"{cluster_name}", ontology_dir)

    print(f"\n✅ Analysis complete! Check results in: {ontology_dir}")

if __name__ == "__main__":
    main()

