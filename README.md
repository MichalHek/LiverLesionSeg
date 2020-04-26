# LiverLesionSeg
Code for liver lesion segmentation using different architectures.

This code was used to train the lesion segmentation network in our paper:
[Hierarchical Fine-Tuning for joint Liver Lesion Segmentation and Lesion Classification in CT](https://arxiv.org/abs/1907.13409).


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
In order to test the trained cascade you should have two trained models: liver model and lesion model.
The general pipeline is illustrated below:
<img src="https://github.com/MichalHek/LiverLesionSeg/blob/master/images/pipeline_detailed.PNG"  width="700"/> 

- To test the liver segmentation model on the liver evaluation set run ```test/eval_liver.py```
- To test the lesion segmentation model on the lesion evaluation set run ```test/eval_lesion.py```

**To test the full cascade results (liver+lesion model):**
- Generate liver crops for testing by running: ```data/generate_liver_crops_test.py``` with your trained liver model (define it in the script).
- Generate niftii files by running ```test/submit_results.py```. 
- To view the results in a 3D view I recommand using [ITK snap](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3).
- Zip files and submit: [LiTS competition](https://competitions.codalab.org/competitions/17094#results).

