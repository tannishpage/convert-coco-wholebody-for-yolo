# Coco-wholebody dataset converter

This repsitory provides a conversion tool and instructions on how to convert the [Coco-wholebody](https://github.com/jin-s13/COCO-WholeBody) dataset and train the YOLO pose model on the dataset.

# Instructions

### Step 1 Clone this repository

Create a virtual environment

#### Using Conda
```
conda create -n yolo_environment

# Activate environment
conda activate yolo_environment
```

#### Using Pyenv
```
python3 -m venv ./yolo_environment

# Activate on Posix
source ./yolo_environment/bin/activate
```

```
git clone https://github.com/tannishpage/convert-coco-wholebody-for-yolo
```

### Step 2 Download Coco-wholebody Annotations and Coco17 Images

On the official github for [Coco-wholebody](https://github.com/jin-s13/COCO-WholeBody/tree/master#:~:text=Google%20Drive:%20COCO%2DWholeBody%20annotations%20for%20Train%20/%20Validation%20(Google%20Drive)) you'll find download links for the train and validation annotation files. Download both and put them into a folder

To download the Coco17 images go to the official [Coco Dataset download](https://cocodataset.org/#download) page for the links or use the below commands.

```
mkdir coco17-images
cd coco17-images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
```

Be sure to unzip both zip files

```
unzip train2017.zip
unzip val2017.zip
```

### Step 3 Run the converter

Now go into the cloned repository and run the python script. Provide the location where the annotation json files were saved, the train and val images folder, and an output location (this where the converted dataset will be placed). 

```
cd convert-coco-wholebody-for-yolo
python3 -m pip install -r requirments.txt
python3 convert_data.py ../coco-wholebody-annotations/ ../coco17-images/train2017/ ../coco17-images/val2017/ ../converted_data/
```

After converting the dataset you can follow the steps from the [Ultralytics Docs](https://docs.ultralytics.com/datasets/pose#usage) to train the model. Be sure to point the `data` parameter to the `.yaml` file located in the output folder.