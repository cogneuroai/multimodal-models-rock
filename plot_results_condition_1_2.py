import base64
import requests
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm



import argparse

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


# Global rocks index for image filename creation


# Function to format the image filename
def create_image_filename(row):
    
    rocks_index = {1: 'I_Andesite', 2: 'I_Basalt', 3: 'I_Diorite', 4: 'I_Gabbro', 5: 'I_Granite', 6: 'I_Obsidian', 7: 'I_Pegmatite', 8: 'I_Peridotite', 9: 'I_Pumice', 10: 'I_Rhyolite', 11: 'M_Amphibolite', 12: 'M_Anthracite', 13: 'M_Gneiss', 14: 'M_Hornfels', 15: 'M_Marble', 16: 'M_Migmatite', 17: 'M_Phyllite', 18: 'M_Quartzite', 19: 'M_Schist', 20: 'M_Slate', 21: 'S_Bituminous Coal', 22: 'S_Breccia', 23: 'S_Chert', 24: 'S_Conglomerate', 25: 'S_Dolomite', 26: 'S_Micrite', 27: 'S_Rock Gypsum', 28: 'S_Rock Salt', 29: 'S_Sandstone', 30: 'S_Shale'}

    rock_type = rocks_index.get(int(row['subtype']), 'Unknown')
    token_str = str(int(row['token within subtype'])).zfill(2)
    return f"{rock_type}_{token_str}.jpg"

def load_and_preprocess_human_data(filepath, columns, image_naming_func, usecols=None, delimiter=None, header=None):
    # Automatically determine the file type and choose the loading method
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        human_data = pd.read_excel(filepath, usecols=usecols, header=None)
    elif filepath.endswith('.csv') or filepath.endswith('.txt'):
        # For TXT files, a common use case is to have whitespace as a delimiter
        if filepath.endswith('.txt'):
            delimiter = delimiter if delimiter is not None else '\s+'
        human_data = pd.read_csv(filepath, header=header, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .txt, or .xlsx file.")
    
    human_data.columns = columns
    human_data = human_data.dropna()
    human_data['Image'] = human_data.apply(image_naming_func, axis=1)
    return human_data

def load_gpt_data(filepaths, expected_columns, directory):
    # Initialize an empty DataFrame with 'Image' column
    combined_data = pd.DataFrame(columns=['Image'])
    combined_data.set_index('Image', inplace=True)
    
    for filepath, new_col_name in zip(filepaths, expected_columns[1:]):  # Skip 'Image', which is common
        # Load the current file
        temp_df = pd.read_csv(directory + filepath)
        
        # Rename 'Response' to the new column name
        temp_df.rename(columns={'Response': new_col_name}, inplace=True)
        temp_df.set_index('Image', inplace=True)
        
        # Merge with the combined DataFrame
        if combined_data.empty:
            combined_data = temp_df
        else:
            combined_data = combined_data.join(temp_df, how='outer')
    
    combined_data.reset_index(inplace=True)
    return combined_data

def intersect_and_save_data(human_data, gpt_data, human_save_path, gpt_save_path):
    columns_intersection = set(human_data.columns).intersection(gpt_data.columns)
    gpt_data_filtered = gpt_data[gpt_data['Image'].isin(human_data['Image'])]
    human_data_filtered = human_data[human_data['Image'].isin(gpt_data['Image'])]

    human_data_intersection = human_data_filtered[list(columns_intersection)]
    gpt_data_intersection = gpt_data_filtered[list(columns_intersection)]

    human_data_intersection.to_csv(human_save_path, index=False)
    gpt_data_intersection.to_csv(gpt_save_path, index=False)

def plot_correlations(csv_path_1, csv_path_2, plot_path, plot_x_label):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # Identify the intersection of columns between the two DataFrames, excluding 'Image' if it exists
    common_columns = set(df1.columns).intersection(df2.columns)
    common_columns.discard('Image')  # Exclude 'Image' column if present
    common_columns = sorted(list(common_columns))

    # Filter both DataFrames to only those common columns
    df1_filtered = df1[common_columns].astype(float)
    df2_filtered = df2[common_columns].astype(float)

    # Plotting setup for square plots
    num_features = len(common_columns)
    num_rows = num_cols = int(np.ceil(np.sqrt(num_features)))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*4, num_rows*4), subplot_kw={'aspect': 'equal'})
    fig.tight_layout(pad=4.0)
    axes_flat = axes.flatten()

    # Hide unused subplots if any
    for i in range(num_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Plot each feature's correlation with ticks set to 1-9 regardless of the data
    for i, feature in enumerate(common_columns):
        x = df1_filtered[feature].values
        y = df2_filtered[feature].values

        # Calculate Pearson correlation
        correlation, _ = pearsonr(x, y)

        # Scatter plot
        axes_flat[i].scatter(x, y, label=f'Corr: {correlation:.2f}')
        axes_flat[i].set_title(f'{feature}')
        axes_flat[i].set_xlabel(plot_x_label)
        axes_flat[i].set_ylabel('Human')

        # Setting ticks from 1 to 9
        axes_flat[i].set_xticks(range(1, 10))
        axes_flat[i].set_yticks(range(1, 10))

        # Optionally, set axis limits to better frame the ticks if necessary
        axes_flat[i].set_xlim(0, 10)
        axes_flat[i].set_ylim(0, 10)

        # Fit line
        m, b = np.polyfit(x, y, 1)
        axes_flat[i].plot(x, m * x + b, color='red', linestyle='--')
        axes_flat[i].legend()

    # plt.savefig(plot_path)
    fig.savefig(plot_path, dpi=fig.dpi)
    # plt.show()


"""
def plot_correlations_combined(csv_path_pairs, plot_path, plot_x_label):
    # Define the fixed dimensions for a 2x5 grid
    num_rows = 2
    num_cols = 5
    
    # Create a large figure to hold all subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
    fig.tight_layout(pad=4.0)
    
    # Adjust for when there's a single subplot
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
        
    axes_flat = axes.flatten()
    plot_index = 0
    
    # Iterate over each pair of CSV files
    for csv_path_1, csv_path_2 in csv_path_pairs:
        df1 = pd.read_csv(csv_path_1)
        df2 = pd.read_csv(csv_path_2)
        
        # Identify common columns, excluding 'Image'
        common_columns = sorted(list(set(df1.columns).intersection(df2.columns) - {'Image'}))
        
        # Filter DataFrames to these columns
        df1_filtered = df1[common_columns].astype(float)
        df2_filtered = df2[common_columns].astype(float)
        
        # Plot each feature in its own subplot
        for feature in common_columns:
            if plot_index >= len(axes_flat):
                break  # Stop if there are more features than subplot spaces

            x = df1_filtered[feature].values
            y = df2_filtered[feature].values
            correlation, _ = pearsonr(x, y)
            ax = axes_flat[plot_index]
            ax.scatter(x, y, label=f'Corr={correlation:.2f}', alpha=0.6)
            ax.set_title(feature)
            ax.set_xlabel(plot_x_label)
            ax.set_ylabel('Human')
            
            # Fit line
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x + b, 'r-')
            
            ax.legend()
            plot_index += 1
    
    # Hide unused subplots
    for k in range(plot_index, len(axes_flat)):
        axes_flat[k].set_visible(False)
    # plt.savefig(plot_path, dpi = fig.dpi)
    fig.savefig(plot_path, dpi=fig.dpi)
"""








def plot_correlations_combined(csv_path_pairs, plot_path, plot_x_label):
    # Set the font properties
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 48  # You can also set the font size here

    # Define the fixed dimensions for a 2x5 grid
    num_rows = 2
    num_cols = 5
    horizontal_spacing = 0.4  # Adjust horizontal spacing here
    vertical_spacing = 0.5    # Adjust vertical spacing here

    # Create a large figure to hold all subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*10, num_rows*9.5))
    
    # Adjust subplot layout
    plt.subplots_adjust(wspace=horizontal_spacing, hspace=vertical_spacing)
    
    # Adjust for when there's a single subplot
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
        
    axes_flat = axes.flatten()
    plot_index = 0
    
    # Fixed ticks
    ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Iterate over each pair of CSV files
    for csv_path_1, csv_path_2 in csv_path_pairs:
        df1 = pd.read_csv(csv_path_1)
        df2 = pd.read_csv(csv_path_2)
        
        # Identify common columns, excluding 'Image'
        common_columns = sorted(list(set(df1.columns).intersection(df2.columns) - {'Image'}))
        
        # Filter DataFrames to these columns
        df1_filtered = df1[common_columns].astype(float)
        df2_filtered = df2[common_columns].astype(float)
        
        # Plot each feature in its own subplot
        for feature in common_columns:
            if plot_index >= len(axes_flat):
                break  # Stop if there are more features than subplot spaces

            x = df1_filtered[feature].values
            y = df2_filtered[feature].values
            correlation, _ = pearsonr(x, y)
            ax = axes_flat[plot_index]
            ax.scatter(x, y)
            ax.set_title(feature, pad=25)

            # Set axis limits and ticks
            ax.set_xlim(0.5, 9.5)
            ax.set_ylim(0.5, 9.5)
            ax.set_aspect('equal')
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            ax.tick_params(axis='both', which='major', direction='inout', length=10, width=2)


            # Modify spines to remove the top and right box borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Fit line
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x + b, 'r-')
            ax.text(1.25, 8.5, f'r = {correlation:.2f}', fontsize=48, color='black')

            # Conditional label setting for Y-axis
            if plot_index % num_cols == 0:  # For plots on the left side in each row
                ax.set_ylabel('Human', labelpad=20)
            else:
                ax.set_ylabel('')

            # Conditional label setting for X-axis
            if plot_index >= num_cols * (num_rows - 1):  # For all plots in the last row
                ax.set_xlabel(plot_x_label, labelpad=20)
            else:
                ax.set_xlabel('')

            plot_index += 1
    
    # Hide unused subplots
    for k in range(plot_index, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    # plt.show()








def continuous(data_path, folder_name, save_path):
    # Load and combine the GPT data from multiple CSV files
    # save_directory = 'Opus_Results/no_anchor/'
    # Human_Save_Path = 'Opus_Results'
    # Gpt_Save_Path = 'Opus_Results'

    
    save_directory = f'{data_path}/{folder_name}/'
    Human_Save_Path = f'{save_path}/{folder_name}/'
    Gpt_Save_Path = f'{save_path}/{folder_name}/'

    create_directory_if_not_exists(f'{save_path}/{folder_name}/')

    if 'haiku' in folder_name:
        plot_x_label = 'Haiku'
    elif 'sonnet' in folder_name:
        plot_x_label = 'Sonnet'
    elif 'opus' in folder_name:
        plot_x_label = 'Opus'
    elif 'gpt4' in folder_name:
        plot_x_label = 'GPT-4'

    # Define the column names for the human data
    human_data_columns = ['subtype', 'token within subtype', 'darkness/lightness', 'fine/coarse grain',
                        'smooth/rough', 'dull/shiny', 'disorganized/organized', 'chromaticity', 'red/green']

    # Load the human data from an Excel file
    human_data = load_and_preprocess_human_data(
        filepath="Data/rocknorm3607_dat_catnumbered.xlsx",
        columns=human_data_columns,
        image_naming_func=create_image_filename,
        usecols=[i for i in range(9)]  # Specify the columns to use from the Excel file
    )

    # Define the specific GPT columns to include
    gpt_columns = ['Image', 'darkness/lightness', 'fine/coarse grain', 'smooth/rough', 'dull/shiny', 'disorganized/organized', 'chromaticity', 'red/green']

    # Specify the list of GPT data files
    gpt_files = [
        'model_lightness.csv', 'model_grain_size.csv', 'model_roughness.csv',
        'model_shininess.csv', 'model_organization.csv', 'model_chromaticity.csv','model_red_green_hue.csv']


    gpt_data = load_gpt_data(gpt_files, gpt_columns, save_directory)

    # Intersect the human and GPT data based on the 'Image' column, then save the intersected data
    intersect_and_save_data(
        human_data=human_data,
        gpt_data=gpt_data,
        human_save_path=f'{Human_Save_Path}/human_continuous_combined.csv',
        gpt_save_path=f'{Gpt_Save_Path}/model_continuous_no_anchors_combined_2.csv'
    )

    # Plot the correlations between the saved datasets
    plot_correlations(
        csv_path_1=f'{Gpt_Save_Path}/model_continuous_no_anchors_combined_2.csv',
        csv_path_2=f'{Human_Save_Path}/human_continuous_combined.csv',
        plot_path =  f"{save_path}/{folder_name}/{folder_name}_continuous.png",
        plot_x_label = plot_x_label
    )


def supplimentary(data_path, folder_name, save_path):
    save_directory = f'{data_path}/{folder_name}/'
    Human_Save_Path = f'{save_path}/{folder_name}/'
    Gpt_Save_Path = f'{save_path}/{folder_name}/'

    create_directory_if_not_exists(f'{save_path}/{folder_name}/')

    # Define the columns for human data
    human_data_columns = ['rock number', 'subtype', 'token within subtype', 'porphyritic texture', 
                        'presence of holes', 'green hue', 'pegmatitic structure', 'conchoidal fracture']


    # Load human data
    human_data = load_and_preprocess_human_data(
        filepath="Data/supp540.txt",
        columns=human_data_columns,
        image_naming_func=create_image_filename
        # No need to specify 'usecols' or 'delimiter' for TXT as we use the defaults for this format
    )

    if 'haiku' in folder_name:
        plot_x_label = 'Haiku'
    elif 'sonnet' in folder_name:
        plot_x_label = 'Sonnet'
    elif 'opus' in folder_name:
        plot_x_label = 'Opus'
    elif 'gpt4' in folder_name:
        plot_x_label = 'GPT-4'

    # Define GPT files and the corresponding columns after 'Image'
    gpt_files = [
        'model_porphyritic_texture.csv',
        'model_pegmatitic_structure.csv',
        'model_conchoidal_fracture.csv'
    ]
    gpt_columns = ['Image', 'porphyritic texture', 'pegmatitic structure', 'conchoidal fracture']

    # Load and process GPT data

    gpt_data = load_gpt_data(gpt_files, gpt_columns, save_directory)

    # Intersect and save the processed data
    intersect_and_save_data(
        human_data=human_data,
        gpt_data=gpt_data,
        human_save_path=f'{Human_Save_Path}/human_supplementary_combined.csv',
        gpt_save_path=f'{Gpt_Save_Path}/model_supplementary_no_anchors_combined_2.csv'
    )

    # Plot correlations
    plot_correlations(
        csv_path_1=f'{Gpt_Save_Path}/model_supplementary_no_anchors_combined_2.csv',
        csv_path_2=f'{Human_Save_Path}/human_supplementary_combined.csv',
        plot_path =  f"{save_path}/{folder_name}/{folder_name}_supplimentary.png",
        plot_x_label = plot_x_label)



def combined_continuous_supplimentary(folder_name, save_path):
    # Only to be run after running continuous and supplimentary functions
    Human_Save_Path_Continuous = f'{save_path}/{folder_name}/human_continuous_combined.csv'
    Gpt_Save_Path_Continuous = f'{save_path}/{folder_name}/model_continuous_no_anchors_combined_2.csv'
    Human_Save_Path_Supplimentary = f'{save_path}/{folder_name}/human_supplementary_combined.csv'
    Gpt_Save_Path_Supplimentary = f'{save_path}/{folder_name}/model_supplementary_no_anchors_combined_2.csv'

    if 'haiku' in folder_name:
        plot_x_label = 'Haiku'
    elif 'sonnet' in folder_name:
        plot_x_label = 'Sonnet'
    elif 'opus' in folder_name:
        plot_x_label = 'Opus'
    elif 'gpt4' in folder_name:
        plot_x_label = 'GPT-4'

    csv_path_pairs = [(Gpt_Save_Path_Continuous, Human_Save_Path_Continuous),
                      (Gpt_Save_Path_Supplimentary, Human_Save_Path_Supplimentary)]
    
    plot_correlations_combined(csv_path_pairs = csv_path_pairs,
                               plot_path=f'{save_path}/{folder_name}/{folder_name}_combined.png',
                               plot_x_label=plot_x_label)
    




if __name__ == '__main__':
    # continuous()
    # supplimentary()

    # python plot_results_rgb.py --data_class='all' --folder_name='3A_haiku_data_no_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3A_sonnet_data_no_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3B_haiku_data_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3C_haiku_data_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3C_haiku_data_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3C_haiku_data_anchor_images'
    # python plot_results_rgb.py --data_class='all' --folder_name='3C_haiku_data_anchor_images'


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_class', choices=['continuous', 'supplimentary', 'all'], required=True)
    parser.add_argument('--data_path', type=str, default='Data_From_LLM_Experiments')
    parser.add_argument('--folder_name', type=str, required = True) # data folder for specific model
    parser.add_argument('--save_path', type=str, default='Plots_From_LLM_Experiments')
    args = parser.parse_args()

    if args.data_class == 'all':
        continuous(data_path= args.data_path,
                   folder_name=args.folder_name,
                   save_path=args.save_path)
        
        supplimentary(data_path= args.data_path,
                      folder_name=args.folder_name,
                      save_path=args.save_path)
        
        combined_continuous_supplimentary(folder_name=args.folder_name, 
                                          save_path=args.save_path)
    elif args.data_class == 'continuous':
        continuous(data_path= args.data_path,
                   folder_name=args.folder_name,
                   save_path=args.save_path)
    elif args.data_class == 'supplimentary':
        supplimentary(data_path= args.data_path,
                      folder_name=args.folder_name,
                      save_path=args.save_path)


