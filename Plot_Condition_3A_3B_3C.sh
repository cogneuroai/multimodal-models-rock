#!/bin/bash

# Path to the text file containing the names
FILE="Data_From_LLM_Experiments/Condition_3A_3B_3C_folder_names.txt"

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "File $FILE does not exist."
    exit 1
fi

# Loop through each line in the file
while IFS= read -r name; do
    # Process each name
    echo "Processing $name"
    python plot_results_condition_3A_3B_3C.py --data_class='all' --folder_name=$name
    
done < "$FILE"
