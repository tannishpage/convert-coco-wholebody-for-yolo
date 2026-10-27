import json
import os
import sys
import numpy as np
import shutil

# What the script needs to convert the data
# Location of the json files
# Location of the images
# Output location

def main():
    if len(sys.argv) != 5:
        print("python3 convert_data.py <location_of_json_file> <location_of_train_images> <location_of_val_images> <output_location>")
        sys.exit()

    print("Setting up folders...", end='')
    
    json_file_location = sys.argv[1]
    image_locs = {}
    image_locs['train'] = sys.argv[2]
    image_locs['val'] = sys.argv[3]
    output_location = sys.argv[4]

    json_configs = [os.path.join(json_file_location, file) for file in os.listdir(json_file_location) if file.endswith(".json") and "coco" in file]

    # Check if the output directory exists
    if not os.path.exists(output_location):
        os.makedirs(output_location)

    # Now we make two sub directories images and labels
    # And within each we make train and validation

    img_train_loc = os.path.join(output_location, "images", "train")
    img_val_loc = os.path.join(output_location, "images", "val")
    label_train_loc = os.path.join(output_location, "labels", "train")
    label_val_loc = os.path.join(output_location, "labels", "val")


    os.makedirs(img_train_loc, exist_ok=True)
    os.makedirs(img_val_loc, exist_ok=True)
    os.makedirs(label_train_loc, exist_ok=True)
    os.makedirs(label_val_loc, exist_ok=True)
        
    print("Done")

    # Read the Json files and copy the images over to the folder structure above
    annotation_keys_to_parse = ['bbox', 'keypoints', 'foot_kpts', 'face_kpts', 'lefthand_kpts', 'righthand_kpts']
    for json_file in json_configs:
        print(f"Converting {os.path.basename(json_file)} to YOLO style...", end='')
        sys.stdout.flush()
        
        labels = json.load(open(json_file))

        if ('images' not in labels) or ('annotations' not in labels):
            print(f"JSON file {os.path.basename(json_file)} does not contain 'images' or 'annotations' please check file")
            sys.exit()

        if ("train" in os.path.basename(json_file)) or ("train" in labels['images'][0]['coco_url']):
            mode = 'train'
        elif ("val" in os.path.basename(json_file)) or ("val" in labels['images'][0]['coco_url']):
            mode = 'val'
        else:
            print(f"Unable to determine mode: train or val. Please check {os.path.basename(json_file)} to ensure it is correct")
            sys.exit()

        img_name_lookup_dict = dict()
        img_to_keypoints = dict()

        # Build image name set without the 000000 so we can look up by keys
        for img in labels['images']:
            file_name = img["file_name"]
            file_id = int(img["file_name"].replace(".jpg", ""))
            height = img['height']
            width = img['width']
            img_name_lookup_dict[file_id] = (file_name, height, width)

        # Build YOLO style keypoint annotations for each image
        for annotation in labels["annotations"]:
            img_id = annotation['image_id']
            img_name, height, width = img_name_lookup_dict[img_id]
            output = [0]
            for key in annotation_keys_to_parse:
                keypoint_array = np.array(annotation[key])

                # Normalize the keypoitns to the image width and height
                if key != 'bbox':
                    keypoint_array[0::3] = keypoint_array[0::3]/width
                    keypoint_array[1::3] = keypoint_array[1::3]/height

                if key == 'bbox':
                    keypoint_array[0] += (keypoint_array[2]/2)
                    keypoint_array[1] += (keypoint_array[3]/2)
                    keypoint_array[0::2] = keypoint_array[0::2]/width
                    keypoint_array[1::2] = keypoint_array[1::2]/height
                output += list(keypoint_array)

            if img_to_keypoints.get(img_name, None) is None:
                img_to_keypoints[img_name] = [output]
            else:
                img_to_keypoints[img_name].append(output)

        for img_name in img_to_keypoints:
            # print(img_name)
            # print(join(img_to_keypoints[img_name][0], " "))
            label_file_name = img_name.replace(".jpg", ".txt")

            # Opening files for writing
            with open(os.path.join(output_location, "labels", mode, label_file_name), 'w') as label_file: 
            
                # Write keypoints to file
                for kpts in img_to_keypoints[img_name]:
                    label_file.write(" ".join(map(str, kpts)) + "\n")

                label_file.flush()

            # Copy the image over
            shutil.copyfile(os.path.join(image_locs[mode], img_name), os.path.join(output_location, "images", mode, img_name))

        print("Done")

    config_yaml_file_contents = f"'path: {os.path.abspath(output_location)} # dataset root dir\ntrain: images/train # train images (relative to root)\nval: images/val     # val images (relative to root)\n\n# Keypoints shape: [number of keypoints, 3] \n# (133 keypoints total, each with x, y, and visibility)\nkpt_shape: [133, 3] \n\n# Classes\nnames:\n  0: person'"
    config_yaml_file = open(os.path.join(output_location, "coco-wholebody.yaml"), 'w')
    config_yaml_file.write(config_yaml_file_contents)
    config_yaml_file.flush()
    config_yaml_file.close()

    # Print file counts in output directories
    print(f"{len(os.listdir(img_train_loc))} files in images/train")
    print(f"{len(os.listdir(img_val_loc))} files in images/val")
    print(f"{len(os.listdir(label_train_loc))} files in label/train")
    print(f"{len(os.listdir(label_val_loc))} files in label/val")
    print(f"Yaml file saved to {os.path.join(output_location, "coco-wholebody.yaml")}")
    print("------- Data Successfully Converted -------")

if __name__ == '__main__':
    main()