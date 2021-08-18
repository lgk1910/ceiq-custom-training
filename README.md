# Training CEIQ model for contrast-distorted image assessment with custom dataset

## Environment
```
conda create -n CEIQ_training python=3.7
pip install -r requirements.txt
```
## Note
The two folders **test_imgs** and **dataset** and the csv file **contrast_distorted_dmos.csv** are used to demonstrate the training process. You can train the model with custom dataset by specifying arguments in the commands.

## Preprocessing
### Requirements
CSV file must contains at least two columns:
1. One column named **img_name** representsimage names.
2. One column named **dmos** represents DMOS (Differential Mean Opinion Score) values, each of which is the mean of opinion score that people give to that particular image.

### Command
Run the following command:
```
python preprocessing -i contrast_distorted_dmos.csv -f dataset/contrast_distorted
```
Arguments:
- -i: name of the input csv file
- -f : relative path of the folder that contains training images.

## Training
Run the following command:
```
python train.py -o CEIQ_model.pickle
```
Arguments:
- -o: name of the output file in which the model will be stored.

## Demo:
The model accepts two kinds of input type:
- Option 0: Predicting score with inputs are paths to images.
- Option 1: Predicting score with inputs are RGB matrix reprentation of images.

Run the following command:
```
python demo.py -m CEIQ_model.pickle
```
Arguments:
- -m: path to the CEIQ model

