# LiverLesionSeg
Code for liver lesion segmentation using different architectures

### Training
- Download the LiTS dataset from [this link](https://drive.google.com/drive/folders/0B0vscETPGI1-eE53ZnA0MGhWZFE).
- Preprocess data by running ```data/preprocess_lits.py```.

We train 2 netwroks- one for liver segmentation and one for lesion segmentation as illustrated below:
<img src="https://github.com/MichalHek/LiverLesionSeg/blob/master/images/pipeline.PNG"  width="700"/> 
- Run liver segmentation by running ```train/train_liver.py.py```. Define training parameters in ```train/liver_config.json```
- Before training lesion segmentation- run the following script: ```LiverLesionSeg/data/generate_liver_crops_train.py``` to generate GT liver ROI crops.

