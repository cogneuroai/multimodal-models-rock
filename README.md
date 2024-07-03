# multimodal models rock 🎸🪨

This repo contains relevant code and data for the experiments described in the paper: 

**Comparing Perceptual Judgements in Large Multimodal Models and Humans**

<!---
[__Comparing Perceptual Judgements in Large Multimodal Models and Humans__](insert psyarxiv link here)

To cite this paper or code, please use:

```
insert bibtex / citation here
```
-->

Interactive plots of individual feature data are available at [cognlp.com](https://cognlp.com)

install requirements
```shell
pip install -r requirements.txt
```

## Generating Rock Ratings Using LLMs
In order to generate rock ratings, assign your OpenAI and/or Anthropic API key to API_KEY_ANTHROPIC and API_KEY_OPENAI variables respectively in the python files starting with 'run_condition'.
### Arguments
Python command line arguments in run_condition_1_2.py (for condition 1 and 2), run_condition_3A_3B.py (for condition 3A and 3B) and run_condition_3C.py (for condition 3C) - 
- --eval_category (str) specifies the categories for which rock ratings will be generated. (Defaults to 'all' which refers to iteratively evaluating all categories available)
    - For condition 1 and 2, the available categories are: "all", "lightness", "grain_size", "roughness", "shininess", "organization", "chromaticity", "red_green_hue", "porphyritic_texture", "pegmatitic_structure", and  "conchoidal_fracture".
    - For condition 3A and 3B, the available categories are: "all", "organization" and "pegmatitic_structure".
    - For condition 3C, the available categories are: "all", "organization" and "pegmatitic_structure"
- --anchor_images (bool) is an argument to include or exclude anchor images. (False for excluding, True for including anchor images. Defaults to False)
- --model_name (str) specifies the model name. ('gpt4', 'haiku', 'sonnet', 'opus')
- --model_temperature (float) controls randomness in the model's output ([Link](https://docs.anthropic.com/en/docs/resources/glossary#temperature) for Anthropic models, [Link](https://platform.openai.com/docs/api-reference/runs#runs-createrun-temperature) for OpenAI. Defaults to 1.0)
- --output_dir (str) Directory where the subdirectory with csv files (generated rock ratings) is stored.

The following commands generate rock ratings for respective models - 
### Condition 1
```shell
# Example for gpt-4
python run_condition_1_2.py --eval_category='all' --anchor_images=False --model_name='gpt4' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"
```
### Condition 2
```shell
# Example for haiku
python run_condition_1_2.py --eval_category='all' --anchor_images=True --model_name='haiku' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"
```
### Condition 3A
```shell
# Example for sonnet
python run_condition_3A_3B.py --eval_category='all' --anchor_images=False --model_name='sonnet' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"
```
### Condition 3B
```shell
# Example for opus
python run_condition_3A_3B.py --eval_category='all' --anchor_images=True --model_name='opus' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"
```
### Condition 3C
```shell
# Example for haiku
python run_condition_3C.py --eval_category='all' --anchor_images=True --model_name='haiku' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"
```

### Note:  
After generating the output CSV files, it may be necessary to parse the sentences to extract the rating. This step is important because the Language Model (LLM) may not always follow the output format exactly. Ensure to verify and adjust the extraction process accordingly to maintain data accuracy and consistency.


## Generating Plots From Ratings
- Move the generated directories for Condition 1 and 2 to 'Data_From_LLM_Experiments' directory, and list the names subdirectories for Condition 1 and 2 in Condition_1_2_folder_names.txt inside Data_From_LLM_Experiments directory.
- Move the generated directories for Condition 3A, 3B and 3C to 'Data_From_LLM_Experiments' directory, and list the names subdirectories for Condition 3A, 3B and 3C in Condition_3A_3B_3C_folder_names.txt inside Data_From_LLM_Experiments directory.

To generate plots for Conditions 1 and 2, run -
```shell
./Plot_Condition_1_2.sh
```

To generate plots for Conditions 3A, 3B and 3C, run -
```shell
./Plot_Condition_3A_3B_3C.sh
```
The generated plots will be stored with the same subdirectory names in the 'Plots_From_LLM_Experiments' directory.