# Her2GrayMap
1. Prerequisites
   This repo depends on the following python packages.
   - mmdetection, pls refer https://github.com/open-mmlab/mmdetection to setup the mmdetection env.
   - shapely
   - openslide
   - faiss
   - numpy  
   - opencv-python-headless/opencv-python
   - pandas

2. Data Preparation
   
   - Step 1, annotate whole slide images (WSI, in .mrxs format) exclude regions, i.e., the standard referenced positive/negative marker regions (usually at the bottom of the WSI). 
      
     this repo uses QuPath as labeling tool (for other labeling tolls, users need to customized annotation load function in DataSet class)
     After labeling excluded regions (the default name of excluded regions is "EXCLUDE"), export annotations to GeoJSON format (pls refer https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html).
     
   - Step 2, crop WSI into small patch images.
   
   - Step 3, extract cell membrane for each patch and do color-deconvolution to compute the gray value.
   
   - Step 4, compute the GrayMap for each WSI
   
3. Train Multi-task classification model
```bash
  python train.py
```

4. Evaluate Model Performance
```bash
  python test.py --eval
```

5. Model Inference

