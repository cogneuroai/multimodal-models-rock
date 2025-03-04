# multimodal models rock 🎸🪨

This repo contains relevant code and data for the experiments described
in the paper:

**Comparing Perceptual Judgements in Large Multimodal Models and
Humans**


Interactive plots of individual feature data are available at
[cognlp.com](https://cognlp.com)

Install requirements -

    pip install -r requirements.txt

'Data\_From\_LLM\_Experiments' directory contains the rock ratings
generated by all models in the paper. 'Plots\_From\_LLM\_Experiments'
contains all the plots.

## Generating Rock Ratings Using LLMs

In order to generate rock ratings, assign your OpenAI and/or Anthropic
API key to API\_KEY\_ANTHROPIC and API\_KEY\_OPENAI variables
respectively in the python files starting with 'run\_condition'.

### Arguments

Python command line arguments in run\_condition\_1\_2.py (for condition
1 and 2), run\_condition\_3A\_3B.py (for condition 3A and 3B) and
run\_condition\_3C.py (for condition 3C) -

-   --eval\_category (str) specifies the categories for which rock
    ratings will be generated. (Defaults to 'all' which refers to
    iteratively evaluating all categories available)
    -   For condition 1 and 2, the available categories are: "all",
        "lightness", "grain\_size", "roughness", "shininess",
        "organization", "chromaticity", "red\_green\_hue",
        "porphyritic\_texture", "pegmatitic\_structure", and
        "conchoidal\_fracture".
    -   For condition 3A and 3B, the available categories are: "all",
        "organization" and "pegmatitic\_structure".
    -   For condition 3C, the available categories are: "all",
        "organization" and "pegmatitic\_structure"
-   --anchor\_images (bool) is an argument to include or exclude anchor
    images. (False for excluding, True for including anchor images.
    Defaults to False)
-   --model\_name (str) specifies the model name. ('gpt4', 'haiku',
    'sonnet', 'opus')
-   --model\_temperature (float) controls randomness in the model's
    output
    ([Link](https://docs.anthropic.com/en/docs/resources/glossary#temperature)
    for Anthropic models,
    [Link](https://platform.openai.com/docs/api-reference/runs#runs-createrun-temperature)
    for OpenAI. Defaults to 1.0)
-   --output\_dir (str) Directory where the subdirectory with csv files
    (generated rock ratings) is stored.

The following commands generate rock ratings for respective models -

### Condition 1

    # Example for gpt-4
    python run_condition_1_2.py --eval_category='all' --anchor_images=False --model_name='gpt4' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"

### Condition 2

    # Example for haiku
    python run_condition_1_2.py --eval_category='all' --anchor_images=True --model_name='haiku' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"

### Condition 3A

    # Example for sonnet
    python run_condition_3A_3B.py --eval_category='all' --anchor_images=False --model_name='sonnet' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"

### Condition 3B

    # Example for opus
    python run_condition_3A_3B.py --eval_category='all' --anchor_images=True --model_name='opus' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"

### Condition 3C

    # Example for haiku
    python run_condition_3C.py --eval_category='all' --anchor_images=True --model_name='haiku' --model_temperature=0 --output_dir="Data_From_LLM_Experiments"

### Note:

After generating the output CSV files, it may be necessary to parse the
sentences to extract the rating. This step is important because the
Language Model (LLM) may not always follow the output format exactly.
Ensure to verify and adjust the extraction process accordingly to
maintain data accuracy and consistency.

## Generating Plots From Ratings

-   Move the generated directories for Condition 1 and 2 to
    'Data\_From\_LLM\_Experiments' directory, and list the names
    subdirectories for Condition 1 and 2 in
    Condition\_1\_2\_folder\_names.txt inside
    Data\_From\_LLM\_Experiments directory.
-   Move the generated directories for Condition 3A, 3B and 3C to
    'Data\_From\_LLM\_Experiments' directory, and list the names
    subdirectories for Condition 3A, 3B and 3C in
    Condition\_3A\_3B\_3C\_folder\_names.txt inside
    Data\_From\_LLM\_Experiments directory.

To generate plots for Conditions 1 and 2, run -

    ./Plot_Condition_1_2.sh

To generate plots for Conditions 3A, 3B and 3C, run -

    ./Plot_Condition_3A_3B_3C.sh

The generated plots will be stored with the same subdirectory names in
the 'Plots\_From\_LLM\_Experiments' directory.
