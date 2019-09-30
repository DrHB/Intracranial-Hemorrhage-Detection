# Intracranial-Hemorrhage-Detection

Intracranial hemorrhage, bleeding that occurs inside the cranium, is a serious health problem requiring rapid and often intensive medical treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient.

Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

In this competition, your challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes.

## Setup
- Convert Ddicom formant to .png. (Since we are dealing with CT scans its important to select window so far I have been using 40/40 for more details check `src/dicom_to_png.py`) -> After conversion png files are `512x512`
- One dicom file was corupted please run `src/00_DICOM_PNG.ipynb` to adjust train dataframe
- We also have to readjust train.csv to match fastai input format for labels using `src/convert_df_to_fastai.ipynb`
- Resize images to `224x224` using `src/resize_pngs_fast.ipynb` for faster testing

# EXP_00
Just intial test with  EfficientNet-B0

### EXP_00.ipynb
```
MODEL:           EfficientNet-B5
NUM_CLASSES:     6
BS:              512
SZ:              224
VALID:           1 FOLD CV (FOLD=0)
TFMS:            get_transform()
PRETRAINED:      IMAGNET
NORMALIZE:       IMAGENET

TRAINING:        OPT: Adam
                 fit_one_cycle(lr=1e-2/3, epoch=5)-Everything Frozen Except Last Layer
                 
                 OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-3/8, epoch=20, decay_start=0.7)-Unfrozen

MODEL WEIGHTS:   [NB_EXP_00_CV_0_UNFRZ.pth, NB_EXP_00_CV_0_PHASE_2_COS.pth]
MODEL TRN_LOSS:  0.054292
MODEL VAL_LOSS:  0.059521
ACCURACY THRES:  0.978539
LB SCORE:        0.086 (SUB_NAME: NB_EXP_00_CV_0.csv)
LB SCORE_TTA:    0.078 (SUB_NAME: NB_EXP_00_CV_0_COS_TTA.csv)
```
Comments: Pretrained model trained just OLD DATA gives pretty good results
