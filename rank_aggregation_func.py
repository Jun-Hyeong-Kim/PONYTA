import pandas as pd
import numpy as np

from tqdm import tqdm

import os

import pyflagr.Weighted as Weighted
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian


''' 
output_path
- output path that has output folders each from single iteration
- Each folder assumed to have gene ranks from 5-fold cross-validation.
'''

def rank_aggregation(output_path, agg_method='weight_dibra'):

    
    list_input = pd.DataFrame()
    qrel_input = pd.DataFrame()
    
    output_folders = [folder for folder in os.listdir(output_path)]
    
    fold_num = 1
    
    for iter_folder in output_folders:
    
        iter_folder_path = os.path.join(output_path, iter_folder)
    
        for fold in range(1,5+1):
    
            fold_file = pd.read_csv(os.path.join(iter_folder_path, f'gene_rank_fold{fold}.csv'))
            gene_col = fold_file['gene']
    
            # input list formation
            qu_col = pd.DataFrame({'query': [1] * fold_file.shape[0]})
            vot_col = pd.DataFrame({'voter':[f'V-{fold_num}']*fold_file.shape[0]})
            dat_col = pd.DataFrame({'dataset':[ko_gene]*fold_file.shape[0]})
            
            fold_file = pd.concat([vot_col, fold_file], axis=1)
            fold_file = pd.concat([qu_col, fold_file], axis=1)
            fold_file = pd.concat([fold_file, dat_col], axis=1)
    
            fold_num += 1
            list_input = pd.concat([list_input, fold_file], axis=0)
            # print(list_input.shape)
    
    ### List input should not have header ###    
    new_headers = list_input.iloc[0]
    
    # Remove the first row from the DataFrame
    list_input = list_input[1:]
    list_input.columns = new_headers

    if agg_method == 'weight_dibra':

        # weighted DIBRA
        
        method = Weighted.DIBRA(gamma=1.5, prune=True, d1=0.3, d2=0.05)
        df_out, df_eval = method.aggregate(input_df=list_input)
        df_out = df_out.drop(['Query', 'ItemID'], axis=1)
        df_out.to_csv(os.path.join(output_path, f'gene_rank_{agg_method}.csv'), index=False)
        
    elif agg_method == 'linear_borda':

        # linear combination with Borda normalization
        
        csum = Linear.CombSUM(norm='borda')
        df_out, df_eval = csum.aggregate(input_df=list_input)
        df_out = df_out.drop(['Query', 'ItemID'], axis=1)
        df_out.to_csv(os.path.join(output_path, f'gene_rank_{agg_method}.csv'), index=False)

    else:
        
        # Majoritarian method
    
        condorcet = Majoritarian.CondorcetWinners()
        df_out, df_eval = condorcet.aggregate(input_df=list_input)
        df_out = df_out.drop(['Query', 'ItemID'], axis=1)
        df_out.to_csv(os.path.join(output_path, f'gene_rank_{agg_method}.csv'), index=False)
        


def rank_aggregation_median(output_path):
    
    output_folders = os.listdir(output_path)
    merged_df = pd.DataFrame()
    
    for output_folder in output_folders:
    
    
        print(f'Start processing on {output_folder}...')
    
        
        rank1 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold1.csv'))
        rank1['rank'] = rank1['pos_prob'].rank(ascending=False, method='average')
        
        rank1.sort_values(by='gene', inplace=True)
    
        
        rank2 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold2.csv'))
        rank2['rank'] = rank2['pos_prob'].rank(ascending=False, method='average')
    
        
        rank2.sort_values(by='gene', inplace=True)
    
        
        rank3 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold3.csv'))
        rank3['rank'] = rank3['pos_prob'].rank(ascending=False, method='average')
    
        
        rank3.sort_values(by='gene', inplace=True)
    
        
        rank4 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold4.csv'))
        rank4['rank'] = rank4['pos_prob'].rank(ascending=False, method='average')
    
        rank4.sort_values(by='gene', inplace=True)
    
        
        rank5 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold5.csv'))
        rank5['rank'] = rank5['pos_prob'].rank(ascending=False, method='average')
    
        rank5.sort_values(by='gene', inplace=True)
    
        merged_df = pd.concat([merged_df, rank1, rank2, rank3, rank4, rank5], ignore_index=True)
    
    
    median_df = merged_df.groupby('gene')['rank'].mean().reset_index().sort_values(by='rank')
    median_df.to_csv(os.path.join(output_path, f'gene_rank_median.csv'), index=False)

def rank_aggregation_avg(output_path):
    
    output_folders = os.listdir(output_path)
    merged_df = pd.DataFrame()
    
    for output_folder in output_folders:
    
    
        print(f'Start processing on {output_folder}...')
    
        
        rank1 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold1.csv'))
        rank1['rank'] = rank1['pos_prob'].rank(ascending=False, method='average')
        
        rank1.sort_values(by='gene', inplace=True)
    
        
        rank2 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold2.csv'))
        rank2['rank'] = rank2['pos_prob'].rank(ascending=False, method='average')
    
        
        rank2.sort_values(by='gene', inplace=True)
    
        
        rank3 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold3.csv'))
        rank3['rank'] = rank3['pos_prob'].rank(ascending=False, method='average')
    
        
        rank3.sort_values(by='gene', inplace=True)
    
        
        rank4 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold4.csv'))
        rank4['rank'] = rank4['pos_prob'].rank(ascending=False, method='average')
    
        rank4.sort_values(by='gene', inplace=True)
    
        
        rank5 = pd.read_csv(os.path.join(output_path, output_folder, 'gene_rank_fold5.csv'))
        rank5['rank'] = rank5['pos_prob'].rank(ascending=False, method='average')
    
        rank5.sort_values(by='gene', inplace=True)
    
        merged_df = pd.concat([merged_df, rank1, rank2, rank3, rank4, rank5], ignore_index=True)
    
    
    avg_df = merged_df.groupby('gene')['rank'].mean().reset_index().sort_values(by='rank')
    avg_df.to_csv(os.path.join(output_path, f'gene_rank_avg.csv'), index=False)

    
