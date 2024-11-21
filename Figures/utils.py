import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_for_cc_modularity(datasets_path:str,
                                 synthetics_path:str,
                                 condition_prefix:str,
                                 reorder:bool=True) -> pd.DataFrame:
    
    npy_files:list = glob.glob(datasets_path+"*.npy")
    _=npy_files.pop(npy_files.index(synthetics_path))
    
    dataframe_list:list = []
    
    for file_path in (npy_files):
        temporary_holder:np.ndarray = np.load(file_path)
        temporary_df:pd.DataFrame = pd.DataFrame(temporary_holder.reshape(1, -1),
                                                 columns=[f'{j/10}' for j in range(temporary_holder.shape[0])])
        
        basename:str = os.path.basename(file_path)
        condition_name:str = basename.replace(condition_prefix, "").replace(".npy", "")
        if "human_struct" in condition_name:
            condition_name = condition_name.replace("human_struct_scale", "human_")
        temporary_df['condition'] = condition_name
        dataframe_list.append(temporary_df)
        
    empirical_datasets:pd.DataFrame = pd.concat(dataframe_list, ignore_index=True)
    
    synthetics:np.ndarray = np.load(synthetics_path)
    synthetics:np.ndarray = synthetics.mean(axis=1)
    n_steps:int = synthetics.shape[1]
    condition_names:list = ['random', 'small_world', 'modular']

    synthetic_datasets:pd.DataFrame = pd.DataFrame(synthetics.reshape(-1, n_steps), columns=[f'{j/10}' for j in range(n_steps)])
    synthetic_datasets['condition'] = condition_names
    datasets:pd.DataFrame = pd.concat([empirical_datasets, synthetic_datasets], ignore_index=True)
    datasets:pd.DataFrame = datasets.melt(id_vars='condition', var_name='reciprocity', value_name=condition_prefix)
    
    if reorder:
        nicer_order:list = [
        'human_033', 'human_060', 'human_125', 'human_250', 'human_500',
        'macaque_modha','rat', 'mouse', 'drosophila', 'celegans', 
        'random', 'small_world', 'modular']
        datasets['condition'] = pd.Categorical(datasets['condition'], categories=nicer_order, ordered=True)
        datasets:pd.DataFrame = datasets.sort_values(by=['condition', 'reciprocity'], ascending=[True, True])
    return datasets


def preprocess_for_distance(datasets_path:str,
                            synthetics_path:str,
                            condition_prefix:str,
                            reorder:bool=True) -> (pd.DataFrame, pd.DataFrame):
    
    def compute_metrics(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
        global_efficiency = np.zeros(matrix.shape[-1])
        diameter = np.zeros_like(global_efficiency)
        reciprocal_matrices = np.reciprocal(matrix, where=matrix!=0)
        for reciprocity in range(matrix.shape[-1]):
            np.fill_diagonal(reciprocal_matrices[:,:,reciprocity], 0)
            global_efficiency[reciprocity] = reciprocal_matrices[:,:,reciprocity].mean()
            finite_values = matrix[:,:,reciprocity][np.isfinite(matrix[:,:,reciprocity])]
            diameter[reciprocity] = finite_values.max()
        return global_efficiency, diameter
    
    def process_file(file_path: str) -> (pd.DataFrame, pd.DataFrame):
        data = np.load(file_path)
        global_efficiency, diameter = compute_metrics(data)
        
        global_efficiency_df = pd.DataFrame(global_efficiency.reshape(1, -1),
                                            columns=[f'{j/10}' for j in range(global_efficiency.shape[0])])
        diameter_df = pd.DataFrame(diameter.reshape(1, -1),
                                   columns=[f'{j/10}' for j in range(diameter.shape[0])])
        
        basename = os.path.basename(file_path)
        condition_name = basename.replace(condition_prefix, "").replace(".npy", "")
        if "human_struct" in condition_name:
            condition_name = condition_name.replace("human_struct_scale", "human_")
        
        global_efficiency_df['condition'] = condition_name
        diameter_df['condition'] = condition_name
        
        return global_efficiency_df, diameter_df
    
    npy_files = glob.glob(datasets_path + "*.npy")
    npy_files.remove(synthetics_path)
    
    global_efficiency_list = []
    diameter_list = []
    
    for file_path in npy_files:
        global_efficiency_df, diameter_df = process_file(file_path)
        global_efficiency_list.append(global_efficiency_df)
        diameter_list.append(diameter_df)
        
    empirical_global_efficiency = pd.concat(global_efficiency_list, ignore_index=True)
    empirical_diameter = pd.concat(diameter_list, ignore_index=True)
    
    synthetics = np.load(synthetics_path)
    synthetics = synthetics.mean(axis=3)
    n_steps = synthetics.shape[-1]
    condition_names = ['random', 'small_world', 'modular']
    
    reciprocal_synthetics = np.reciprocal(synthetics, where=synthetics!=0)
    global_efficiency_synthetics = np.zeros((reciprocal_synthetics.shape[-2], reciprocal_synthetics.shape[-1]))
    diameter_synthetics = np.zeros_like(global_efficiency_synthetics)
    
    for reciprocity in range(reciprocal_synthetics.shape[-1]):
        for condition in range(len(condition_names)):
            np.fill_diagonal(reciprocal_synthetics[:,:,condition,reciprocity], 0)
            global_efficiency_synthetics[condition,reciprocity] = reciprocal_synthetics[:,:,condition,reciprocity].mean()
            diameter_synthetics[condition,reciprocity] = reciprocal_synthetics[:,:,condition,reciprocity].max()
    
    synthetic_global_efficiency_df = pd.DataFrame(global_efficiency_synthetics.reshape(-1, n_steps),
                                                  columns=[f'{j/10}' for j in range(n_steps)])
    synthetic_diameter_df = pd.DataFrame(diameter_synthetics.reshape(-1, n_steps),
                                         columns=[f'{j/10}' for j in range(n_steps)])
    
    synthetic_global_efficiency_df['condition'] = condition_names
    synthetic_diameter_df['condition'] = condition_names
    
    global_efficiency_datasets = pd.concat([empirical_global_efficiency, synthetic_global_efficiency_df], ignore_index=True)
    diameter_datasets = pd.concat([empirical_diameter, synthetic_diameter_df], ignore_index=True)
    
    global_efficiency_datasets = global_efficiency_datasets.melt(id_vars='condition', var_name='reciprocity', value_name=f'{condition_prefix}_global_efficiency')
    diameter_datasets = diameter_datasets.melt(id_vars='condition', var_name='reciprocity', value_name=f'{condition_prefix}_diameter')
    
    if reorder:
        nicer_order = [
            'human_033', 'human_060', 'human_125', 'human_250', 'human_500',
            'macaque_modha', 'rat', 'mouse', 'drosophila', 'celegans', 
            'random', 'small_world', 'modular'
        ]
        global_efficiency_datasets['condition'] = pd.Categorical(global_efficiency_datasets['condition'], categories=nicer_order, ordered=True)
        global_efficiency_datasets = global_efficiency_datasets.sort_values(by=['condition', 'reciprocity'], ascending=[True, True])
        
        diameter_datasets['condition'] = pd.Categorical(diameter_datasets['condition'], categories=nicer_order, ordered=True)
        diameter_datasets = diameter_datasets.sort_values(by=['condition', 'reciprocity'], ascending=[True, True])
    
    return global_efficiency_datasets, diameter_datasets



def plot_scatter_and_kde(data:pd.DataFrame,
                  x:str,
                  y:str,
                  axes:list[plt.Axes, plt.Axes, plt.Axes],
                  scatter_kwargs:dict=None,
                  kde_kwargs:dict=None)->plt.Axes:
    scatter_kwargs:dict = scatter_kwargs if scatter_kwargs else {}
    kde_kwargs:dict = kde_kwargs if kde_kwargs else {}

    sns.kdeplot(data=data,x=x,ax=axes[0],**kde_kwargs)
    sns.scatterplot(data=data,x=x, y=y,ax=axes[1],**scatter_kwargs)
    sns.kdeplot(data=data,y=y,ax=axes[2],**kde_kwargs)
    
    axes[0].sharex(axes[1])
    axes[2].sharey(axes[1])
    
    axes[0].set_yticks([])
    axes[2].set_xticks([])

    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    axes[0].tick_params(labelbottom=False)
    axes[2].tick_params(labelleft=False)
    sns.despine(ax=axes[0],left=True)
    sns.despine(ax=axes[1])
    sns.despine(ax=axes[2],left=False,bottom=True)