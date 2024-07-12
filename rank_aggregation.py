import pandas as pd
import numpy as np

from tqdm import tqdm

import os


''' 
output_path
- output path that has output folders each from single iteration
- Each folder assumed to have gene ranks from 5-fold cross-validation.
'''

def rank_aggregation(output_path):
    
    output_folders = os.listdir(output_path)
    
    for output_folder in output_folders:
    
    
        print(f'Start processing on {output_folder}...')
    
        
        rank1 = pd.read_csv(os.path.join(output_path,output_folder, 'gene_rank_fold1.csv'))
        rank1['rank'] = rank1['pos_prob'].rank(ascending=False, method='average')
        
        rank1.sort_values(by='gene', inplace=True)
    
        
        rank2 = pd.read_csv(os.path.join(output_path,output_folder, 'gene_rank_fold2.csv'))
        rank2['rank'] = rank2['pos_prob'].rank(ascending=False, method='average')
    
        
        rank2.sort_values(by='gene', inplace=True)
    
        
        rank3 = pd.read_csv(os.path.join(output_path,output_folder, 'gene_rank_fold3.csv'))
        rank3['rank'] = rank3['pos_prob'].rank(ascending=False, method='average')
    
        
        rank3.sort_values(by='gene', inplace=True)
    
        
        rank4 = pd.read_csv(os.path.join(output_path,output_folder, 'gene_rank_fold4.csv'))
        rank4['rank'] = rank4['pos_prob'].rank(ascending=False, method='average')
    
        rank4.sort_values(by='gene', inplace=True)
    
        
        rank5 = pd.read_csv(os.path.join(output_path,output_folder, 'gene_rank_fold5.csv'))
        rank5['rank'] = rank5['pos_prob'].rank(ascending=False, method='average')
    
        rank5.sort_values(by='gene', inplace=True)
    
        merged_df = pd.concat([merged_df, rank1, rank2, rank3, rank4, rank5], ignore_index=True)
    
    
    avg_df = merged_df.groupby('gene')['rank'].mean().reset_index().sort_values(by='rank')
    avg_df.to_csv(os.path.join(output_path, output_folder, f'gene_rank_avg.csv'), index=False)
    
