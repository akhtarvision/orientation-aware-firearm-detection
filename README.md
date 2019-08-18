# Orientation Aware Object Detection with Applications to Firearms

Automatic detection of firearms is important for enhancing security and safety of people, however, it is a challenging task owing to the wide variations in shape, size and appearance of firearms. Viewing angle variations and occlusions by the weapon’s carrier and the surrounding people, further increases the difficulty of the task. Moreover, the existing object detectors process rectangular areas, though a thin and long rifle may actually cover only a small percentage of that area and the rest may contain irrelevant details suppressing the required object signatures. To handle these challenges we propose an Orientation Aware Object Detector (OAOD) which has achieved improved firearm detection and localization performance.

![alt text](https://github.com/makhtar17004/orientation-aware-firearm-detection/blob/master/images/flow_diagram_web.jpg)



# Instructions

This code is moderated using Faster RCNN. We made the two phases in Faster RCNN by adopting cascade approach. Please see the setup details of Faster RCNN [here](https://github.com/rbgirshick/py-faster-rcnn). This will assist in runnig our model.

We provide necessaery files to run test script using our model. Download our model from this [link](https://drive.google.com/file/d/1ShZoCTfoBga9j0y-GPINOFgdf1x8Ti9t/view?usp=sharing). Put it into .../data/faster_rcnn_models directory.

Replace the cfg, test in fast rcnn folder. Also replace the prototxt file for test with the provided one. Also put images in .../data/demo folder.


After installation and setup, to run the test file. Place it into .../tools directory:
```python demo_firearms.py```




# Results:

![alt text](https://github.com/makhtar17004/orientation-aware-firearm-detection/blob/master/images/more_results_web.jpg)


# Paper and Model Link

Here is the arXiv link: https://arxiv.org/abs/1904.10032

Here is the web-link: http://im.itu.edu.pk/orientation-aware-firearms-detection/

Trained model: [link](https://drive.google.com/file/d/1ShZoCTfoBga9j0y-GPINOFgdf1x8Ti9t/view?usp=sharing)

# DATASET

[DATASET is available upon request [Google Form]](https://forms.gle/t3dS5g5JQdfPoSvn9)



BIBTEX:

```@article{DBLP:journals/corr/abs-1904-10032,
  author    = {Javed Iqbal and
               Muhammad Akhtar Munir and
               Arif Mahmood and
               Afsheen Rafaqat Ali and
               Mohsen Ali},
  title     = {Orientation Aware Object Detection with Application to Firearms},
  journal   = {CoRR},
  volume    = {abs/1904.10032},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.10032}
}
```


