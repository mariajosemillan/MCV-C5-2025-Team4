import os
import json

def create_messages_json(folder_path, output_file):
    messages = []

    image_extensions = ('.jpg')
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    #prompt3#{"type": "text", "text": "Give a recipe title describing the food in the image. If no title is found give an empty string."},
                    #{"type": "text", "text": "generate exactly 1 title for the recipe/food/drink based on its key ingredients. Each title should be clear, catchy, and descriptive, highlighting the most important or unique ingredients in the dish. For example, if the key ingredients are potatoes and seasoning, a title might be 'Crispy Salt and Pepper Potatoes'. For a dish with mac and cheese and Thanksgiving flavors, a title could be 'Thanksgiving Mac and Cheese'. You can also include elements that reflect the style or theme of the recipe (e.g., 'Italian Sausage and Bread Stuffing'). Return only the title."},
                    {"type": "text", "text": "Give 1 phrase describing the food in the image from its key ingredients."},
                ],
            }
            
            messages.append(message)
    
    with open(output_file, 'w') as f:
        json.dump(messages, f, indent=4)
    
    print(f"Successfully created {output_file} with {len(messages)} image entries.")

folder_path = "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/food_dataset_split/test"
output_file = "messages_QWEN3.json"
create_messages_json(folder_path, output_file)