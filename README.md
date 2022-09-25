# Her2GrayMap
1. Prerequisites
   This repo depends on mmdetection, pls refer https://github.com/open-mmlab/mmdetection to setup the mmdetection env.

2. Data Preparation
   
   - Step 1, annotate whole slide images (WSI, in .mrxs format) exclude regions, i.e., the standard referenced positive/negative marker regions (usually at the bottom of the WSI). 
      
   - Step 2, crop WSI into small patch images.
   
   - Step 3, extract cell membrane for each patch and do color-deconvolution to compute the gray value.
   
   - Step 4, compute the GrayMap for each WSI
   
3. Train Multi-task classification model
   

4. Evaluate Model Performance


5. Model Inference
