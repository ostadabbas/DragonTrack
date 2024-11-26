import json
import os
import base64

def encode_image_to_base64(image_path):
    """
    Encode the image at the given path to base64.
    """
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_line(line):
    """
    Parse a line from the input file and extract relevant data.
    Adjusts to use top-left and bottom-right corner format for bounding boxes.
    """
    parts = line.strip().split(',')
    frame_id, subject_id, xmin, ymin, width, height = map(int, parts[:6])
    # Calculate the bottom-right corner
    xmax = xmin + width
    ymax = ymin + height
    
    return frame_id, {
        "label": str(subject_id),
        "points": [[xmin, ymin], [xmax, ymax]],  # top-left and bottom-right corners
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "mask": None
    }

def process_input_file(input_file_path, images_directory, output_directory):
    """
    Process the input file, organizing data by frame, and create a JSON file for each frame,
    including Base64-encoded image data.
    """
    frames_data = {}
    with open(input_file_path, 'r') as file:
        for line in file:
            frame_id, shape = parse_line(line)
            if frame_id not in frames_data:
                frames_data[frame_id] = []
            frames_data[frame_id].append(shape)
    
    for frame_id, shapes in frames_data.items():
        image_path = os.path.join(images_directory, f"frame{frame_id}.jpg")
        image_data = encode_image_to_base64(image_path) if os.path.exists(image_path) else None
        base_json = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"frame{frame_id}.jpg",
            "imageData": image_data,
            "imageHeight": 794,  # Adjust as needed
            "imageWidth": 794    # Adjust as needed
        }
        output_file_path = os.path.join(output_directory, f"frame{frame_id}.json")
        with open(output_file_path, 'w') as output_file:
            json.dump(base_json, output_file, indent=4)
        print(f"JSON data for frame {frame_id} has been saved to {output_file_path}")

# Paths
input_file_path = 'MOT17-08-FRCNN.txt'  # Adjust this to the path of your input file
images_directory = '/home/galoaa.b/ondemand/data'  # Adjust to where your images are stored
output_directory = 'output_jsons/'   # Adjust this to your desired output directory

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process the input file and generate JSON files
process_input_file(input_file_path, images_directory, output_directory)