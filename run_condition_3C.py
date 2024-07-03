import base64
import requests
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from anthropic import Anthropic
import argparse
import time
import random
from distutils.util import strtobool
import requests

from types import SimpleNamespace
from openai import OpenAI



# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string


# Query model
def query_model_API(image_path, prompt_text, API_KEY, anchor_images_info = None, model_name = 'claude-3-haiku-20240307', model_temperature = 1.0):
    if model_name.startswith('claude'):
        client = Anthropic(api_key=API_KEY)
    else:
        # client = OpenAI(api_key=API_KEY)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    base64_image = encode_image(image_path)

    if anchor_images_info is None:
        if model_name.startswith('claude'):
            message_input= [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "text", "text": f"This is the rock you will label."},
                        {"type": "image", 
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": f"{base64_image}"
                            }
                            }
                        ] 
                }
            ]
        else:
            message_input= [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "text", "text": f"This is the rock you will label."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}] 
                }
            ]
    else:
        
        base64_anchors = [(encode_image(img_path), label) for img_path, label in anchor_images_info]
        message_content = []
        message_content.append({"type": "text", "text": prompt_text})
        for img, label in base64_anchors:
            message_content.append({"type": "text", "text": f"This is an example of a {label} rock."})
            if model_name.startswith('claude'):
                message_content.append({"type": "image", "source": {"type": "base64","media_type": "image/jpeg","data": f"{img}"}})
            else:
                message_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{img}"})
                
        message_content.append({"type": "text", "text": "This is the rock you will label."})
        
        if model_name.startswith('claude'):
            message_content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg","data": f"{base64_image}"}})
        else:
            message_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"})

        message_input = [
            {
                "role": "user",
                "content": message_content
            }
        ]

    for attempt in range(0,10):
        try:
            if model_name.startswith('claude'):    
                response = client.messages.create(
                    model = model_name,
                    messages = message_input,
                    max_tokens = 300,
                    temperature = model_temperature
                )
            else:

                payload = {
                    "model": model_name,
                    "messages": message_input,
                    "max_tokens": 300,
                    "temperature": model_temperature}
                
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response = response.json()
                


        except:
            print(f"attempt - {attempt} failed, The server threw and error we will try again in 15 seconds")
            time.sleep(15)
        else:
            # executes only when try clause works perfectly
            break
    if model_name.startswith('claude'):
        return response.content #json()
    else:
        return [SimpleNamespace(text=response['choices'][0]['message']['content'])]





def process_images(prompt_text, csv_file_path, image_directory, model_name, model_temperature, API_KEY, anchor_images_info= None, limit=0):
    try:
        df_responses = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        df_responses = pd.DataFrame(columns=['Image', 'Response'])
    # print(df_responses)
    # exit()
    image_paths = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(image_directory) for file in files if file.endswith(".jpg")]
    image_paths.sort()

    # image_paths = image_paths[-4:]

    # print(len(image_paths))
    # for index, val in enumerate(image_paths):
    #     if 'S_Rock Gypsum_06.jpg' in val:
    #         print(index)
    #         break

    # exit()
    if limit > 0:
        image_paths = image_paths[:limit]

    for image_path in tqdm(image_paths, desc="Processing images"):
        row_label = os.path.basename(image_path)

        if row_label not in df_responses['Image'].values:
            response = query_model_API(image_path= image_path, 
                                       prompt_text= prompt_text,
                                       API_KEY= API_KEY, 
                                       anchor_images_info= anchor_images_info, 
                                       model_name = model_name,
                                       model_temperature = model_temperature)
            try:
                parsed_response = response[0].text #response['choices'][0]['message']['content']
            except KeyError:
                print(response)
                break

            row_data = {'Image': row_label, 'Response': parsed_response}
            temp_df = pd.DataFrame([row_data])
            df_responses = pd.concat([df_responses, temp_df], ignore_index=True)

            # Save progress after each image is processed
            df_responses.to_csv(csv_file_path, index=False)

    print(f"Processing complete. Responses saved to {csv_file_path}")


def filter_csv_files(directory_path):
    """
    Iterates through all CSV files in the given directory,
    filtering out rows where the 'Response' column contains 'sorry'.
    Each file is then saved back to its original location.
    """
    # Check if the provided directory path exists and is a directory
    if not os.path.isdir(directory_path):
        print(f"The path {directory_path} is not a valid directory.")
        return
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Filter the DataFrame
                df_filtered = df[~df['Response'].str.contains('sorry', case=False, na=False)]
                
                # Save the filtered DataFrame back to the file
                df_filtered.to_csv(file_path, index=False)
                
                print(f"Filtered and saved {filename}")
                
            except Exception as e:
                continue

# directory_path = 'gpt_data_no_anchors/'
# filter_csv_files(directory_path)


def _prompt_(eval_category, anchor_images, csv_folder_path, model_name, model_temperature, API_KEY):
    all_categories = ["lightness", "grain_size", "roughness", "shininess", "organization", "chromaticity", "red_green_hue", "porphyritic_texture", "pegmatitic_structure", "conchoidal_fracture", "all"]
    
    
    assert all(elem in all_categories for elem in eval_category), "some mentioned categories do not exist"
    if len(eval_category)>1 and 'all' in eval_category:
        print("If you wish to run all, just say all, instead of a list of categories along with 'all'")
        exit()
    
    if anchor_images == False:
        print("Not implemented!")
        exit()
    else:
        print(f"{anchor_images=}")
        prompt_dict ={
            
            "organization": 'Some rocks have highly organized global textures that are very regular and orderly, yielding systematic structured patterns such as stripes, bands, or physical layers. Other rocks have very disorganized global textures, such as those with fragments or crystals that seem glued together in haphazard fashion. In this trial, you will rate a rock on how disorganized or organized its global texture appears to be. A rock that has very disorganized global textures should receive ratings of 1.00 or 2.00. A rock that has very organized global textures should receive ratings of 8.00 or 9.00. A rock that is medium in its global-texture organization or that has no visible texture should receive medium ratings. Example rocks for different values on the scale 1.00-9.00 are shown. Please use these helper rocks to make your rating. Please try to use the full scale from 1.00 (most disorganized) to 9.00 (most organized) in making your ratings. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "pegmatitic_structure": 'In this experiment you will be presented with a picture of a rock. Certain rocks have long and thick shiny crystal bands that are embedded in a SEPARATE dull background. These long and thick shiny crystal bands are often (but not always) colored black or green. Your job in this experiment is to rate the extent to which the rock shown has the above-described property. Please remember that rocks with thin stripes that are not shiny do NOT have this property and that rocks that are shiny all over do NOT have this property.   The rock needs to have long and thick shiny crystal bands that are embedded in a SEPARATE dull background. Rocks that have nothing like this property should receive ratings between 1.00 or 2.00. Rocks that have a hint of this property should receive ratings between 4.00, 5.00 and 6.00. Rocks that strongly display this property should receive ratings between 8.00 and 9.00. Example rocks for different values on the scale 1.00-9.00 are shown. Please use these helper rocks to make your rating. Please try to use the full scale from 1.00 through 9.00 in making your ratings. Note: Because this property is very rare, most of the time your response will be between 1.00 and 2.00. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.'
        }

    #TODO: Find jpg images
    anchor_images_info_dict = {
        "organization":[("Images_RGB/I_Granite_08.jpg", "4.375"),
                        ("Images_RGB/M_Gneiss_06.jpg", "8.5"),
                        ("Images_RGB/M_Schist_04.jpg", "4.75"),
                        ("Images_RGB/M_Slate_12.jpg", "7.75"),
                        ("Images_RGB/S_Breccia_02.jpg", "1"),
                        ("Images_RGB/S_Conglomerate_08.jpg", "1.625"),
                        ("Images_RGB/S_Dolomite_12.jpg", "6"),
                        ("Images_RGB/S_Rock Salt_04.jpg", "1"),
                        ("Images_RGB/S_Sandstone_06.jpg", "8.625")], 
        
        "pegmatitic_structure":[("Images_RGB/I_Andesite_10.jpg", "3.00"),
                                ("Images_RGB/I_Diorite_02.jpg", "4.00"),
                                ("Images_RGB/I_Pegmatite_07.jpg", "7.88"),
                                ("Images_RGB/I_Pegmatite_08.jpg", "8.00"),
                                ("Images_RGB/I_Pegmatite_11.jpg", "8.00"),
                                ("Images_RGB/M_Schist_11.jpg", "4.50"),
                                ("Images_RGB/S_Conglomerate_04.jpg", "4.50"),
                                ("Images_RGB/S_Dolomite_12.jpg", "1.00"),
                                ("Images_RGB/S_Sandstone_07.jpg", "1.00")]
        }

    if eval_category[0] == "all":
        for i in prompt_dict:
            if anchor_images==False:
                print(f"Evaluating {i} without anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv",
                               image_directory='Images_RGB',
                               anchor_images_info=None, 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY = API_KEY)
            else:
                print(f"Evaluating {i} with anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv",
                               image_directory='Images_RGB',
                               anchor_images_info= anchor_images_info_dict[i], 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY = API_KEY)
    else:
        for i in eval_category:
            if anchor_images==False:
                print(f"Evaluating {i} without anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv", 
                               image_directory='Images_RGB',
                               anchor_images_info=None, 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY = API_KEY)
            else:
                print(f"Evaluating {i} with anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv",
                               image_directory='Images_RGB',
                               anchor_images_info= anchor_images_info_dict[i], 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY = API_KEY)

def string_or_list(s):
    if ',' in s:
        return [i.strip() for i in s.split(',')]
    else:
         return [s]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

def generate_random_id():
    timestamp = int(time.time() * 1000)  # Convert current time to milliseconds
    random_num = random.randint(0, 1000000)  # Generate a random number
    unique_id = (timestamp + random_num) % 1000000  # Take the last 6 digits
    return unique_id



if __name__ == "__main__":
    
    # Add your API keys here
    API_KEY_ANTHROPIC = ""
    API_KEY_OPENAI = ""


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval_category', type=string_or_list, help='categories to evaluate', default='all')
    parser.add_argument('--anchor_images', type=lambda x: bool(strtobool(x)), help='categories to evaluate', default=False)
    parser.add_argument('--model_name', type=str, help='Name of the model to use', choices=['haiku', 'sonnet', 'opus', 'gpt4'], required=True)
    parser.add_argument('--model_temperature', type=float, help='controls the randomness of the model, defaults to 1', default=1.0)
    parser.add_argument('--output_dir', type=str, help='Directory to save the output files', default = "Data_From_LLM_Experiments")
    args = parser.parse_args()

    if args.model_name == 'haiku':
        model_name = 'claude-3-haiku-20240307'
        api_key = API_KEY_ANTHROPIC
    elif args.model_name == 'sonnet':
        model_name = 'claude-3-sonnet-20240229'
        api_key = API_KEY_ANTHROPIC
    elif args.model_name == 'opus':
        model_name = 'claude-3-opus-20240229'
        api_key = API_KEY_ANTHROPIC
    elif args.model_name == 'gpt4':
        model_name = 'gpt-4-vision-preview'
        api_key = API_KEY_OPENAI

    if args.anchor_images==False:
        csv_folder_path = os.path.join(args.output_dir, f"{args.model_name}_data_no_anchor_images_{str(generate_random_id())}")
    else:
        csv_folder_path = os.path.join(args.output_dir, f"{args.model_name}_data_anchor_images_{str(generate_random_id())}")
    
    create_directory_if_not_exists(csv_folder_path)

    
    _prompt_(eval_category = args.eval_category, 
             anchor_images = args.anchor_images,
             csv_folder_path = csv_folder_path,
             model_name = model_name,
             model_temperature = args.model_temperature,
             API_KEY = api_key)
    
