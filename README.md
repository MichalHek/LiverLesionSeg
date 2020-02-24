# LiverLesionSeg
Code for liver lesion segmentation using different architectures

### Training
- Download the LiTS dataset from [this link](https://drive.google.com/drive/folders/0B0vscETPGI1-eE53ZnA0MGhWZFE).
- Intstall *segmentation-models* package from [this link](https://github.com/qubvel/segmentation_models).
- Preprocess data by running ```data/preprocess_lits.py```.

We train 2 netwroks- one for liver segmentation and one for lesion segmentation as illustrated below:
<img src="https://github.com/MichalHek/LiverLesionSeg/blob/master/images/pipeline.PNG"  width="700"/> 
- Run liver segmentation by running ```train/train_liver.py.py```. Define training parameters in ```train/liver_config.json```
- Before training lesion segmentation run the following script: ```data/generate_liver_crops_train.py``` to generate GT liver ROI crops.
- Run lesion segmentation by running ```train/train_lesion.py.py```. Define training parameters in ```train/lesion_config.json```

### Testing
- To test the liver segmentation model run ```
In order to test the trained cascade you should have two trained models: liver model and lesion model.
The general pipeline is illustrated below:
<img src="https://github.com/MichalHek/LiverLesionSeg/blob/master/images/pipeline_detailed.PNG"  width="700"/> 
- Generate liver crops for testing by running: ```data/generate_liver_crops_test.py``` with your trained liver model (define it in the script).
- 

