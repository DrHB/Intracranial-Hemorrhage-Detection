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
MODEL:           EfficientNet-B0
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

### EXP_10.ipynb

Experiment 10 is divided in to 3 sub experiments:
- `EXP_10` Simple Training of xresnet50 with Simple Attential from scratch
- `EXP_10_TFL` reusing weights from EXP_10 to train on img size `512`
- `EXP_10_MIXUP` training from scratch to test mixup

#### EXP_10
```
MODEL:           xresnet50+Attn
NUM_CLASSES:     6
BS:              384
SZ:              224
VALID:           1 FOLD CV (FOLD=0)
TFMS:            get_transform()
PRETRAINED:      False
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-3/8, epoch=20, decay_start=0.7)-Unfrozen

MODEL WEIGHTS:   [NB_EXP_10_CV_0_PHASE_2_COS.pth]
MODEL TRN_LOSS:  0.041329
MODEL VAL_LOSS:  0.055582
ACCURACY THRES:  0.980156
LB SCORE:        0.090 (SUB_NAME: NB_EXP_10_CV_0_COS.csv)
LB SCORE_TTA:    0.079 (SUB_NAME: NB_EXP_10_CV_0_COS_TTA.csv)
```
Comments: Results are very similar to other networks 



#### EXP_10_TFL
```
MODEL:           xresnet50+Attn
NUM_CLASSES:     6
BS:              64
SZ:              512
VALID:           1 FOLD CV (FOLD=0)
TFMS:            get_transform()
PRETRAINED:      False
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-3, epoch=5, decay_start=0.7)-Unfrozen
               
WEIGHTS:         NB_EXP_10_CV_0_PHASE_2_COS - Used to Init 

MODEL WEIGHTS:   [NB_EXP_10_TFL_512_CV_0_PHASE_2_COS.pth]
MODEL TRN_LOSS:  0.050205 	
MODEL VAL_LOSS:  0.055050
ACCURACY THRES:  0.979957
LB SCORE:        0.076 (SUB_NAME: NB_EXP_10_CV_0_TFL_512_COS.csv)
LB SCORE_TTA:    0.074 (SUB_NAME: NB_EXP_10_CV_0_TFL_512_COS_TTA.csv)
```
Comment: Its seems like larger image size trained from the weihgts for images `224` show good LB score 



### EXP_20.ipynb
 ```
 MODEL:           Res2Net50
 NUM_CLASSES:     6
 BS:              384
 SZ:              224
 VALID:           1 FOLD CV (FOLD=0)
 TFMS:            get_transform()
 PRETRAINED:      False
 NORMALIZE:       Data

 TRAINING:        OPT: Radam
                  Policy: Cosine Anneal 
                  flattenAnneal(lr=1e-3/8, epoch=20, decay_start=0.7)-Unfrozen

 MODEL WEIGHTS:   [NB_EXP_20_CV_0_PHASE_2_COS.pth]
 MODEL TRN_LOSS:  0.053704
 MODEL VAL_LOSS:  0.059312
 ACCURACY THRES:  0.978431
 LB SCORE:        0.084 (SUB_NAME: NB_EXP_20_CV_0_COS.csv)
 LB SCORE_TTA:    0.079 (SUB_NAME: NB_EXP_20_CV_0_COS_TTA.csv)
 ```
 Comments: Training from scratch shows similar results
