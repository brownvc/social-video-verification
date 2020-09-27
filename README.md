# Towards Untrusted Social Video Verification to Combat Deepfakes via Face Geometry Consistency

[Eleanor Tursman](https://tursmanor.github.io/),
[Marilyn George](https://cs.brown.edu/people/grad/mgeorge5/),
[Seny Kamara](http://cs.brown.edu/~seny/),
[James Tompkin](http://jamestompkin.com/)  
Brown University  
Media Forensics CVPR Workshop 2020  

<img src="./main-fig.svg" width="70%">

### [Paper + Presentation](http://visual.cs.brown.edu/socialvideoverification/)

## Citation
If you find our work useful for your research, please cite:  

```
@InProceedings{Tursman_2020_CVPR_Workshops,
author = {Tursman, Eleanor and George, Marilyn and Kamara, Seny and Tompkin, James},
title = {Towards Untrusted Social Video Verification to Combat Deepfakes via Face Geometry Consistency},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
} 
```

The functions ```cpca```, ```screeplot```, ```mahalanobis```, ```kernelEVD```, ```greatsort```, and ```classSVD``` are from the [LIBRA toolbox](https://github.com/mwgeurts/libra), which we pulled from their repo at commit 2e1c400e953caf3daa3037305cbd5df09b14bde3 in Feb. 2020, as described in the following papers:  
```
@article{verboven2005libra,
  title={LIBRA: a MATLAB library for robust analysis},
  author={Verboven, Sabine and Hubert, Mia},
  journal={Chemometrics and intelligent laboratory systems},
  volume={75},
  number={2},
  pages={127--136},
  year={2005},
  publisher={Elsevier}
}
``` 
```
@article{verboven2010matlab,
  title={Matlab library LIBRA},
  author={Verboven, Sabine and Hubert, Mia},
  journal={Wiley Interdisciplinary Reviews: Computational Statistics},
  volume={2},
  number={4},
  pages={509--515},
  year={2010},
  publisher={Wiley Online Library}
}
```
LipGAN is described in the following paper:
```
@inproceedings{KR:2019:TAF:3343031.3351066,
  author = {K R, Prajwal and Mukhopadhyay, Rudrabha and Philip, Jerin and Jha, Abhishek and Namboodiri, Vinay and Jawahar, C V},
  title = {Towards Automatic Face-to-Face Translation},
  booktitle = {Proceedings of the 27th ACM International Conference on Multimedia}, 
  series = {MM '19}, 
  year = {2019},
  isbn = {978-1-4503-6889-6},
  location = {Nice, France},
   = {1428--1436},
  numpages = {9},
  url = {http://doi.acm.org/10.1145/3343031.3351066},
  doi = {10.1145/3343031.3351066},
  acmid = {3351066},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {cross-language talking face generation, lip synthesis, neural machine translation, speech to speech translation, translation systems, voice transfer},
}
```
### Running the Code
#### Data Pre-Processing
This section will explain how to process your own data with our scripts. If you'd like to use our data, which includes the raw video and the post-processed landmark matrices, please see [Downloading our Dataset](#downloading-our-dataset).  

##### Requirements
- You will need to download the dlib face detector and put it in `./Data-Processing/` before running the bash script. You can download it [here](http://dlib.net/files/mmod_human_face_detector.dat.bz2). 
- You will also need to install [FAN](https://github.com/1adrianb/face-alignment), by running `pip install face-alignment`. 
- You will need to install the requirements for [LipGAN](https://github.com/Rudrabha/LipGAN). Since it is under an MIT License, I have included the code in this repo, minus their models, which you will need to download yourself and place in `./LipGAN/logs/`. I made modifications to their `batch_inference.py` script so that it would use our pre-computed bounding boxes, and to add a little blending to the final result to make it look a bit nicer. We use the LipGAN repo's commit 03b540c68aa5ab871baa4e64f6ade6736131b1b9, which we pulled Feb 11th, 2020.     

##### Using your own computed landmarks
If you are using your own data, the format the experiment code expects is a struct with entries `cam1`,`cam2`,`cam3`,`cam4`,`cam5`,`cam6`, and `fake`, where the fake is a manipulated version of one of the real cameras. Each field contains an `f x 40` matrix, where there are `f` frames, and each row contains mouth landmarks `[x_1, y_1, x_2, y_2, ..., x_20,  y_20]` (though you may use any subset of landmarks you like--- the 20 mouth landmarks are what we used for all our experiments). Each `f x 40` matrix should be normalized by subtracting the mean and dividing by the standard deviation. We generate our .mat files using `./Data-Processing/landmark-npy-to-mat.py`. Here is an example of how to run that script to generate one of the mat files:  
`python3 landmark-npy-to-mat.py 4 ID6 /path/to/dataset/`  
where the fake camera is camera four, and we're looking at person six from the dataset.   

##### Using our video processing pipe   
The process for pre-processing the data is outlined in `data-pipeline.bash`. To run it:  
`bash -i data-pipeline <path-to-video-folder> <path-to-this-script-folder> <audio filename for lipgan>`. 

The bash script's steps are as follows:  
1. Convert the video to frames using ffmpeg.
2. Get bounding box coordinates for the face in each frame using dlib. Save the results to a txt file. If you want to save the cropped images proper, swap in `saveCrop = True` on line 62 of `cnn_face_detector.py`. Dlib needs to be enabled with CUDA support for this to run reasonably fast. 
3. Run 2D landmark detection using [FAN](https://github.com/1adrianb/face-alignment) given the saved bounding boxes, and save npy files with 68 landmarks per frame. 
4. Create a visualization of the landmarks on top of every camera view, to verify everything has worked properly. 
5. Run LipGAN to create a fake for each input camera, using the audio you specify. The audio must be pre-processed as directed in LipGAN using their matlab scripts before this step, and placed in `./LipGAN/audio/`. (It isn't automated in the bash script.) The bash script expects that you've made a conda environment called 'lipgan' for running lipgan. 
6. Processes the lipgan fakes as in steps 1-3. 

#### Experiments
Before running experimental code, you will have to go into the Matlab function ```clusterdata.m```, and change line one from ```function [T] = clusterdata(X, varargin)``` to ```function [T,Z] = clusterdata(X, varargin)```. Z is the actual tree which we will cluster ourselves.  

- [Full Sequence Fake Detection](#full-sequence-fake-detection)
- [Sliding Window Fake Detection](#sliding-window-fake-detection)
- [Angular Robustness](#angular-robustness)
- [3D Model Experiment](#3d-model-experiment)

##### Full Sequence Fake Detection
To generate the results, make sure you populate your Data file with the normalized landmark matrices. Then run the script `./Experiments/full_sequence_accuracy_experiment.m`.

##### Sliding Window Fake Detection
To generate the results, make sure you populate your Data file with the normalized landmark matrices. Then pick your method on line 10 and run the script `./Experiments/windowed_accuracy_experiment.m`.  

To re-create the accuracy and ROC plots of Fig. 5, set `accOn = true;` and `rocOn = true;` in `Experiments/make_plots.m`. In the paper, we display results for the DWT baseline and for our method. To re-create the histograms in Fig. 6, set `histogramOn = true;`. In the paper, we display results for our method.     
 
For output created using:

- The simple mouth baseline, use `datasetName = 'simpleMouth';`
- The DWT baseline, use `datasetName = noPCA;`
- The PCA method, use `datasetName = onlyPCA;` 

The numbers for Table 1 were generated by looking at the numerical output for the mean accuracy and ROC curve numbers.   

##### Angular Robustness
This experiment tests whether all sets of cameras separated by the same real-world angular distance are more similar to one another than to a fake video. To run, run ```Experiments/angular_experiment.m```. If you do not want to use Matlab's CPU threading, edit line LINE from `parfor t=1:length(threshes)` to `for t=1:length(threshes)`.   

To visualize your results (re-creating Fig. 7), set `angle = true;`, all other flags to false, and `datasetName = angle;` in `./Experiments/make_plots.m`, then run the script.   

##### 3D Model Experiment
Script to re-create Fig. 4 is `./Experiments/3d_model_experiment.py`. It expects the parameter data to be structured like in our dataset, where FLAME and 3DMM parameters are stored in npy files. An example of running the code:  
`python3 3d_model_experiment.py 3 ID5 /path/to/dataset/`  
where the fake camera is camera three, and we're looking at person five from the dataset.   

### Downloading our Dataset
Our video data and post-processed landmarks are available [via the Brown library](https://repository.library.brown.edu/studio/collections/id_1006/).  

To reproduce our results, please use the [post-processed landmark .mat files](https://repository.library.brown.edu/studio/item/bdr:1144738/) with the experimental scripts on Github. Once the .zip is downloaded, extract the contents of `Dataset/Processed-Results/` into `./Experiments/Data/` from the codebase root directory.

### Licensing
All original work is under the GNU General Public License v3.0. LipGAN is under the MIT License, and the LIBRA MATLAB functions are under the academic license specified by LIBRA_LICENSE.

### Changelog
- 2020 August 26th: Moved document downloads to [separate webpage](http://visual.cs.brown.edu/socialvideoverification/). Added errata from paper, thank to [Harman Suri](https://www.linkedin.com/in/harman-suri-487bab127/).
- 2020 August 26th: Added dataset URL.
- 2020 July 15th: Uploaded the code for running our experiments and processing our data. 

