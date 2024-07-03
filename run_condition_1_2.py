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

    image_paths = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(image_directory) for file in files if file.endswith(".jpg")]
    image_paths.sort()

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
        prompt_dict ={
            "lightness": 'In this trial, you will rate a rock on its darkness/lightness of color. A dark rock should receive a rating of 1.00 or 2.00. A light rock should receive a rating of 8.00 or 9.00. A rock that is medium in darkness/lightness should receive a medium rating. In some cases, parts of the rock may be light and other parts may be dark. In those cases, do your best to rate the "average" lightness of the entire rock. Please try to use the full scale from 1.00 (darkest) through 9.00 (lightest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "grain_size": 'In this trial, you will rate a rock on its average grain size. Rocks with no visible grain should receive a rating of 1.00 or 2.00. Rocks with an extremely coarse and fragmented grain should receive a rating of 8.00 or 9.00. Rocks with a medium grain should receive medium ratings. In some cases, parts of the rock may have a fine grain and other parts may have a coarse grain. In those cases, do your best to rate the "average" grain size of the entire rock. Please try to use the full scale from 1.00 (no visible grain) through 9.00 (very coarse grain) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "roughness": 'In this trial, you will rate a rock on how smooth versus rough it appears to be. Rocks that appear to be very smooth should receive a rating of 1.00 or 2.00. Rocks that appear to be very rough should receive a rating of 8.00 or 9.00. Rocks that are medium in their smoothness/roughness should receive medium ratings. In some cases, parts of a rock may be smooth and other parts may be rough. In those cases, do your best to rate the "average" roughness of the entire rock. Please try to use the full scale from 1.00 (smoothest) through 9.00 (roughest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "shininess": 'An object is "shiny" if it reflects light and is glossy. Note that dark-colored objects can still be shiny. In this trial, you will rate a rock on how dull versus shiny it appears to be. Rocks that appear to be very dull should receive a rating of 1.00 or 2.00. Rocks that appear to be very shiny and glossy should receive a rating of 8.00 or 9.00. Rocks that are medium in their dullness/shininess should receive medium ratings. In some cases, parts of a rock may be dull and other parts may be shiny. In those cases, do your best to rate the "average" shininess of the entire rock. Please try to use the full scale from 1.00 (dullest) through 9.00 (shiniest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "organization": 'Some rocks have components that are very regular and organized, such as systematic layers, bands, or grains. Other rocks seem very disorganized, such as those with fragments that are glued together in haphazard fashion. In this trial, you will rate a rock on how disorganized versus organized it appears to be. Rocks that are very disorganized should receive a rating of 1.00 or 2.00. Rocks that are very organized should receive a rating of 8.00 or 9.00. Rocks that are medium in their organization, or that have no visible texture to rate, should receive medium ratings. Please try to use the full scale from 1.00 (most disorganized) through 9.00 (most organized) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "chromaticity": 'In this trial, you will rate a rock in terms of whether it has no color, cool color, or warm color. Rocks with no color (absolute black, gray or white) should receive a rating of 1.00 or 2.00. Rocks with cool colors (blue, blue/green, and green) should receive medium ratings (4.00, 5.00, or 6.00). Rocks with very warm colors (yellow, orange, red) should receive ratings of 8.00 or 9.00. Please try to use the full scale from 1.00 (no color) through 9.00 (warmest color) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "red_green_hue": 'In this experiment, you will be presented with a picture of a rock. We would like you to rate the rock picture on a red-green contrast. Rocks that are most strongly red should receive ratings of 1.00 or 2.00. Rocks that are most strongly green should receive ratings of 8.00 or 9.00. Neutral rocks (black or white) that are absent of color should receive ratings of 5.00. For the remaining rocks, just decide whether the main color tends to be closer to red versus green. For example, most would agree that orange is closer to red than to green, so you might give orange rocks ratings of 2.00, 3.00, or 4.00. Likewise, most would agree that blue is closer to green than to red, so you might give blue rocks ratings of 6.00, 7.00, or 8.00. Please try to use the full scale from 1.00 to 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "porphyritic_texture": "In this experiment you will be presented with a picture of a rock. We are interested in your judgments about a very specific property of some of the rocks -- Certain kinds of rocks contain small fragments or pebbles that are glued into a separate background texture. THESE SMALL FRAGMENTS OR PEBBLES ARE SEPARATE FROM THE REST OF THE ROCK'S BACKGROUND ITSELF. We want you to rate each rock picture for this property. Rocks with no small fragments or pebbles glued into their separate background should receive a rating of 1.00 or 2.00. Rocks that definitely have small fragments or pebbles glued into their separate background should receive a rating of 8.00 or 9.00. Many rocks may be unclear cases; Some may have a coarse grain throughout, but don't really have separate small fragments glued into them. Other rocks may also be hard to judge because they have changes in shading that are not really separate glued fragments. These unclear cases should receive ratings of 4.00, 5.00 or 6.00. Please try to use the full scale from 1.00 through 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.",

            "pegmatitic_structure": 'In this experiment you will be presented with a picture of a rock. Certain rocks have very large-sized crystals that are embedded in a SEPARATE background. The crystals will often (but not always) appear as large shiny bands. Your job in this experiment is simply to judge the extent to which the rock shown in each picture has this property. Rocks that have nothing like this property should receive ratings of 1.00 or 2.00. Rocks that have a hint of this property should receive ratings of 4.00, 5.00, or 6.00. Rocks that strongly display this property should receive ratings of 8.00 or 9.00. Please try to use the full scale from 1.00 through 9.00 in making your rating. Note: Because this property is very rare, most of the time your response will be between 1.00 and 2.00. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "conchoidal_fracture": 'In this experiment, you will be presented with a picture of a rock. We are interested in your judgments about a very specific property of some rocks. The property is called CONCHOIDAL FRACTURES. Conchoidal fractures are formed when pieces of a brittle rock chip off and leave behind smooth, curved surfaces resembling the inside of a seashell. Conchoidal fractures are typically found in glassy or fine-grained rocks. In this trial of the experiment, you will be shown a rock picture. We want you to rate the rock picture for the extent to which it has conchoidal fractures. Rocks with flat or jagged surfaces, or rocks with no fractures should receive a rating of 1.00 or 2.00. Rocks with smooth, curved indents or fractures resembling the inside of a seashell should receive a rating of 8.00 or 9.00. Many rocks may be unclear cases: Some rocks may have fractures where pieces of the rock were chipped off, but they may not be as smooth or curved as true conchoidal fractures. Other rocks may also be hard to judge because they have changes in shading or color. These unclear cases should receive ratings of 4.00, 5.00, or 6.00. Ratings of 8.00 or 9.00 should be given only for rocks for which you are absolutely sure they have conchoidal fractures. Most rocks do not have conchoidal fractures and should receive low ratings. Please try to use the full scale from 1.00 through 9.00 in making your ratings. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.'
        }

        if model_name=='gpt-4-vision-preview':
            check_list = ["grain_size", "red_green_hue", "porphyritic_texture", "pegmatitic_structure", "conchoidal_fracture"]
            prompt_dict = {item:(prompt_dict[item] + " DO NOT RESPOND WITH 'I'm sorry...'" if item in check_list else prompt_dict[item]) for item in prompt_dict}
        
    else:
        print(f"{anchor_images=}")
        prompt_dict ={
            "lightness": 'In this trial, you will rate a rock on its darkness/lightness of color. A dark rock should receive a rating of 1.00 or 2.00. A light rock should receive a rating of 8.00 or 9.00. A rock that is medium in darkness/lightness should receive a medium rating. An example of a very dark rock, a rock that is medium in darkness/lightness, and a very light rock is shown. In some cases, parts of the rock may be light and other parts may be dark. In those cases, do your best to rate the "average" lightness of the entire rock. Please try to use the full scale from 1.00 (darkest) through 9.00 (lightest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "grain_size": 'In this trial, you will rate a rock on its average grain size. Rocks with no visible grain should receive a rating of 1.00 or 2.00. Rocks with an extremely coarse and fragmented grain should receive a rating of 8.00 or 9.00. Rocks with a medium grain should receive medium ratings. An example of a rock with no visible grain, with a medium grain, and with a very coarse and fragmented grain is shown. In some cases, parts of the rock may have a fine grain and other parts may have a coarse grain. In those cases, do your best to rate the "average" grain size of the entire rock. Please try to use the full scale from 1.00 (no visible grain) through 9.00 (very coarse grain) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "roughness": 'In this trial, you will rate a rock on how smooth versus rough it appears to be. Rocks that appear to be very smooth should receive a rating of 1.00 or 2.00. Rocks that appear to be very rough should receive a rating of 8.00 or 9.00. Rocks that are medium in their smoothness/roughness should receive medium ratings. An example of a rock that is very smooth, that is medium in smoothness/roughness, and that is very rough is shown. In some cases, parts of a rock may be smooth and other parts may be rough. In those cases, do your best to rate the "average" roughness of the entire rock. Please try to use the full scale from 1.0 (smoothest) through 9.0 (roughest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "shininess": 'An object is "shiny" if it reflects light and is glossy. Note that dark-colored objects can still be shiny. In this trial, you will rate a rock on how dull versus shiny it appears to be. Rocks that appear to be very dull should receive a rating of 1.00 or 2.00. Rocks that appear to be very shiny and glossy should receive a rating of 8.00 or 9.00. Rocks that are medium in their dullness/shininess should receive medium ratings. An example of a rock that is very dull, that is medium in dullness/shininess, and that is very shiny is shown. In some cases, parts of a rock may be dull and other parts may be shiny. In those cases, do your best to rate the "average" shininess of the entire rock. Please try to use the full scale from 1.00 (dullest) through 9.00 (shiniest) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "organization": 'Some rocks have components that are very regular and organized, such as systematic layers, bands, or grains. Other rocks seem very disorganized, such as those with fragments that are glued together in haphazard fashion. In this trial, you will rate a rock on how disorganized versus organized it appears to be. Rocks that are very disorganized should receive a rating of 1.00 or 2.00. Rocks that are very organized should receive a rating of 8.00 or 9.00. Rocks that are medium in their organization, or that have no visible texture to rate, should receive medium ratings. An example of a rock that is very disorganized, that is medium in organization, and that is highly organized is shown. Please try to use the full scale from 1.00 (most disorganized) through 9.00 (most organized) in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "chromaticity": 'In this trial, you will rate a rock in terms of whether it has no color, cool color, or warm color. Rocks with no color (absolute black, gray or white) should receive a rating of 1.00 or 2.00. Rocks with cool colors (blue, blue/green, and green) should receive medium ratings (4.00, 5.00, or 6.00). Rocks with very warm colors (yellow, orange, red) should receive ratings of 8.00 or 9.00. An example of a rock with no color, cool color, and warm color variation is shown. Please try to use the full scale from 1.00 (no color) through 9.00 (warmest color) in making your rating. Please try to use the full scale from 1.00 to 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "red_green_hue": 'In this experiment, you will be presented with a picture of a rock. We would like you to rate the rock picture on a red-green contrast. Rocks that are most strongly red should receive ratings of 1.00 or 2.00. Rocks that are most strongly green should receive ratings of 8.00 or 9.00. Neutral rocks (black or white) that are absent of color should receive ratings of 5.00. Examples of these different cases are shown. For the remaining rocks, just decide whether the main color tends to be closer to red versus green. For example, most would agree that orange is closer to red than to green, so you might give orange rocks ratings of 2.00, 3.00, or 4.00. Likewise, most would agree that blue is closer to green than to red, so you might give blue rocks ratings of 6.00, 7.00, or 8.00. Please try to use the full scale from 1.00 to 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "porphyritic_texture": "In this experiment you will be presented with a picture of a rock. We are interested in your judgments about a very specific property of some of the rocks -- Certain kinds of rocks contain small fragments or pebbles that are glued into a separate background texture. THESE SMALL FRAGMENTS OR PEBBLES ARE SEPARATE FROM THE REST OF THE ROCK'S BACKGROUND ITSELF. We want you to rate each rock picture for this property. Rocks with no small fragments or pebbles glued into their separate background should receive a rating of 1.00 or 2.00. Rocks that definitely have small fragments or pebbles glued into their separate background should receive a rating of 8.00 or 9.00. Many rocks may be unclear cases; Some may have a coarse grain throughout, but don't really have separate small fragments glued into them. Other rocks may also be hard to judge because they have changes in shading that are not really separate glued fragments. These unclear cases should receive ratings of 4.00, 5.00 or 6.00. Examples of these different cases are shown. Please try to use the full scale from 1.00 through 9.00 in making your rating. Please try to use the full scale from 1.00 to 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.",

            "pegmatitic_structure": 'In this experiment you will be presented with a picture of a rock. Certain rocks have very large-sized crystals that are embedded in a SEPARATE background. The crystals will often (but not always) appear as large shiny bands. Your job in this experiment is simply to judge the extent to which the rock shown in each picture has this property. Rocks that have nothing like this property should receive ratings of 1.00 or 2.00. Rocks that have a hint of this property should receive ratings of 4.00, 5.00, or 6.00. Rocks that strongly display this property should receive ratings of 8.00 or 9.00. Examples of these different cases are shown. Please try to use the full scale from 1.00 through 9.00 in making your rating. Note: Because this property is very rare, most of the time your response will be between 1.00 and 2.00. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.',

            "conchoidal_fracture": 'In this experiment, you will be presented with a picture of a rock. We are interested in your judgments about a very specific property of some rocks. The property is called CONCHOIDAL FRACTURES. Conchoidal fractures are formed when pieces of a brittle rock chip off and leave behind smooth, curved surfaces resembling the inside of a seashell. Conchoidal fractures are typically found in glassy or fine-grained rocks. In this trial of the experiment, you will be shown a rock picture. We want you to rate the rock picture for the extent to which it has conchoidal fractures. Rocks with flat or jagged surfaces, or rocks with no fractures should receive a rating of 1.00 or 2.00. Rocks with smooth, curved indents or fractures resembling the inside of a seashell should receive a rating of 8.00 or 9.00. Many rocks may be unclear cases: Some rocks may have fractures where pieces of the rock were chipped off, but they may not be as smooth or curved as true conchoidal fractures. Other rocks may also be hard to judge because they have changes in shading or color. These unclear cases should receive ratings of 4.00, 5.00, or 6.00. Ratings of 8.00 or 9.00 should be given only for rocks for which you are absolutely sure they have conchoidal fractures. Examples of these different cases are shown. Most rocks do not have conchoidal fractures and should receive low ratings. Please try to use the full scale from 1.00 through 9.00 in making your rating. Please respond ONLY with a continuous numeric decimal value, allowing for any decimal places within the range of 1.00 to 9.00. Precision is key, and values should NOT be constrained to 0.05 increments. Your response can include any decimal point to the hundredths place within the specified range.'
        }

    #TODO: Find jpg images
    anchor_images_info_dict = {
        "lightness": [("Anchors/Low Lightness.jpg", "dark"),
                      ("Anchors/Medium Lightness.jpg", "medium"),
                      ("Anchors/High Lightness.jpg", "light")], 
        
        "grain_size": [("Anchors/Low Grain Size.jpg", "low grain size"),
                       ("Anchors/Medium Grain Size.jpg", "medium grain size"),
                       ("Anchors/High Grain Size.jpg", "high grain size")], 
        
        "roughness":[("Anchors/Low Roughness.jpg", "low roughness"),
                     ("Anchors/Medium Roughness.jpg", "medium roughness"),
                     ("Anchors/High Roughness.jpg", "high roughness")], 
        
        "shininess":[("Anchors/Low Shininess.jpg", "low shininess"),
                     ("Anchors/Medium Shininess.jpg", "medium shininess"),
                     ("Anchors/High Shininess.jpg", "high shininess")], 
        
        "organization":[("Anchors/Low Regularity.jpg", "low organization"),
                        ("Anchors/Medium Regularity.jpg", "medium organization"),
                        ("Anchors/High Regularity.jpg", "high organization")], 
        
        "chromaticity": [("Anchors/Warm Color.jpg", "warm color"),
                         ("Anchors/Cool Color2.jpg", 'cool color'),
                         ("Anchors/No Color2.jpg", "no color")], 
        
        "red_green_hue": [("Anchors/green.jpg", "green"),
                          ("Anchors/neutral_redgreen.jpg", "neutral"),
                          ("Anchors/red.jpg", "red")], 
        
        "porphyritic_texture": [("Anchors/high_porphyritic.jpg", "high porphyritic texture"),
                                ("Anchors/unclear_porphyritic.jpg", 'unclear porphyritic texture'),
                                ("Anchors/low_porphyritic.jpg", "low porphyritic texture")], 
        
        "pegmatitic_structure":[("Anchors/strong_pegmatite.jpg", "high pegmatitic structure"),
                               ("Anchors/medium_pegmatite.jpg", 'unclear pegmatitic structure'),
                               ("Anchors/none_pegmatite.jpg", "low pegmatitic structure")], 
        
        "conchoidal_fracture": [("Anchors/high_conchoidal.jpg", "high conchoidal fracture"),
                                ("Anchors/unclear_conchoidal.jpg", 'unclear conchoidal fracture'),
                                ("Anchors/low_conchoidal.jpg", "low conchoidal fracture")]
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
                               API_KEY=API_KEY)
            else:
                print(f"Evaluating {i} with anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv",
                               image_directory='Images_RGB',
                               anchor_images_info= anchor_images_info_dict[i], 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY=API_KEY)
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
                               API_KEY=API_KEY)
            else:
                print(f"Evaluating {i} with anchor images")
                process_images(prompt_text=prompt_dict[i], 
                               csv_file_path=f"{csv_folder_path}/model_{i}.csv",
                               image_directory='Images_RGB',
                               anchor_images_info= anchor_images_info_dict[i], 
                               limit=0,
                               model_name = model_name,
                               model_temperature = model_temperature,
                               API_KEY=API_KEY)

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
    parser.add_argument('--model_name', type=str, help='Name of the model to use', choices=['haiku', 'sonnet', 'opus','gpt4'], required=True)
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


