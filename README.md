# Intracranial-Hemorrhage-Detection

Intracranial hemorrhage, bleeding that occurs inside the cranium, is a serious health problem requiring rapid and often intensive medical treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient.

Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

In this competition, your challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes.

## Summary 

| Name                  | Network        | Pretrained       | Add             | Img | Img_Processing   | Accuracy | LB    | LB_TTA| Comments |
| -------------         | -------------  | -------------    | -------------   |----|------------      | ---------|  -----|  -----| -------- | 
| EXP_00                | EfficientNetB0 | True             |                 | 224| 40/80                 | 0.978539 |0.086  | 0.078 |          |
| EXP_10                | xresnet50      | False            | Attention       | 224| 40/80                 | 0.980156 |0.090  | 0.079 |          |
| EXP_10_TFL            | xresnet50      | False            | Attention       | 512| 40/80                 | 0.979957 |0.076  | 0.074 |          |
| EXP_10_MIXUP_TFL_512  | xresnet50      | False            | Attention+Mixup | 224| 40/80                 | 0.978396 |0.082  | 0.073 |          |
| EXP_10_MIXUP          | xresnet50      | False            | Attention+Mixup | 224| 40/80                 | 0.979283 |0.084  | 0.074 | Same as above but trained a bi longer |
| EXP_10_MIXUP_TFL_512  | xresnet50      | True             | Attention+Mixup | 512| 40/80                 | 0.980179 |0.077  | 0.072 | Trained on higher image size and did transfer learning  |
| EXP_20                | Res2Net50      | False            |                 | 224| 40/80                 | 0.978431 |0.084  | 0.079 |  |
| EXP_30                | Resnext50      | True             |                 | 224| 40/80                 | 0.980641 |0.095  | 0.079 | added cutout, zoom_rand=1.4 |
| EXP_40                | xresnet50      | True             | Attention       | 224| 40/80, 80/200, 200/450                 | 0.980348 |0.083  | 0.074 |  3 channel diffrent windows, background substractued, trained using `EXP_10_MIXUP` weights|
| EXP_50                | EfficientNetB3      | True           |   | 300| 40/80, 80/200, 200/450  | 0.980348 |0.083  | 0.074 | |

## Setup
- Convert Ddicom formant to .png. (Since we are dealing with CT scans its important to select window so far I have been using 40/40 for more details check `src/dicom_to_png.py`) -> After conversion png files are `512x512`
- One dicom file was corupted please run `src/00_DICOM_PNG.ipynb` to adjust train dataframe
- We also have to readjust train.csv to match fastai input format for labels using `src/convert_df_to_fastai.ipynb`
- Resize images to `224x224` using `src/resize_pngs_fast.ipynb` for faster testing

### EXP_00
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

#### EXP_10_MIXUP
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
                 flattenAnneal(lr=1e-2/2, epoch=25, decay_start=0.7)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_10_CV_0_MIXUP_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.092150 	
MODEL VAL_LOSS:  0.060023 	
ACCURACY THRES:  0.978396
LB SCORE:        0.082 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_COS.csv)
LB SCORE_TTA:    0.073 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_COS_TTA.csv)
```
Comment: Training is super slow, Maybe do 20 more epochs (see below)

```
TRAINING:        OPT: Adap
                 Policy: One Cycle
                 fit_one_cycle(30, lr =1e-3/2)
           
MODEL WEIGHTS:   [NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL.pth]
MODEL TRN_LOSS:  0.086897 	
MODEL VAL_LOSS:  0.057901 	
ACCURACY THRES:  0.979283
LB SCORE:        0.084 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL.csv)
LB SCORE_TTA:    0.074 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL_TTA.csv)
```
Bassicly slow training, improvement are very small... maybe need to train more... good for diversity later on

#### EXP_10_MIXUP_TFL_512
just some finutning using images 512 and with mixup and attention. Weith used for inititlization `NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL` 

```
TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-2/2, epoch=25, decay_start=0.7)-Unfrozen
     
MODEL WEIGHTS:   [NB_EXP_10_CV_0_MIXUP_TFL_512_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.086444 	
MODEL VAL_LOSS:  0.057901 	
ACCURACY THRES:  0.054722
LB SCORE:        0.077 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_TFL_512_PHASE_2_1CYL.csv)
LB SCORE_TTA:    0.072 (SUB_NAME: NB_EXP_10_CV_0_MIXUP_TFL_512_PHASE_2_1CYL_TTA.csv)
```
Training stop att the epoch 12. I did not continue training further 

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



### EXP_30.ipynb
 ```
 MODEL:           resnext50_32x4d
 NUM_CLASSES:     6
 BS:              512
 SZ:              224
 VALID:           1 FOLD CV (FOLD=0)
 TFMS:            get_transform() + [max_zoom = 1.4, xtra_tfms=cutout(n_holes=(4,20), length=(2, 30), p=0.5))]
 PRETRAINED:      True
 NORMALIZE:       Imagenet

 TRAINING:       OPT: Adam
                 fit_one_cycle(lr=1e-2, epoch=10)-Everything Frozen Except Last Layer
                 
                 OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-4, epoch=30, decay_start=0.7)-Unfrozen

 MODEL WEIGHTS:   [NB_EXP_30_CV_0_224_PHASE_1_COS.pth]
 MODEL TRN_LOSS:  0.045051
 MODEL VAL_LOSS:  0.054926
 ACCURACY THRES:  0.980641
 LB SCORE:        0.095 (SUB_NAME: NB_EXP_30_CV_0_224_COS.csv)
 LB SCORE_TTA:    0.079 (SUB_NAME: NB_EXP_30_CV_0_224_COS_TTA.csv)
 ```
 Comments: Here I tried to add cutout augmentation and traind a bit longer to see if this has some effects 
 
 
 ### DATA processing 
So far I have been processing data using `window center =40` , `window height = 80` (check for more detauls 'src/dicom_to_png.py') Unfortenelty it seems like this is not optimal window (see more info: https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing) it seems like for some kind of brain damage the best windowns are `window center =80` , `window height = 200`. Since when doctor check he goes thru diffrent rages. In order to mimic human experience also incorporate windo information I will do following. Make 3 channel RGB Image with 3 diffrent windo sizes `40/80`, `80/200` and very wide `200/450`. When doing this I also notice that there is also a lot of black backround. To remove black background and convert images to 3 channel from dicom files use script `src/dicom_to_png_3chn_bg.py`

EXP_40 will be done using 3 channel images with background substraction. I will be using wights `NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL` from the `EXP_10_MIXUP`


<img src="https://github.com/DrHB/Intracranial-Hemorrhage-Detection/blob/master/src/processing_example.png" width="800">



#### EXP_40
```
MODEL:           xresnet50+Attn
NUM_CLASSES:     6
BS:              384
SZ:              224
VALID:           1 FOLD CV (FOLD=0)
TFMS:            get_transform()
PRETRAINED:      True (NB_EXP_10_CV_0_MIXUP_PHASE_2_1CYL)
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=1e-3, epoch=25, decay_start=0.7)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_40_CV_0_TFL_224_BGS_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.045853 	
MODEL VAL_LOSS:  0.054660	
ACCURACY THRES:  0.980348
LB SCORE:        0.083 (SUB_NAME: NB_EXP_40_CV_0_TFL_224_BGS_PHASE_2_COS.csv)
LB SCORE_TTA:    0.074 (SUB_NAME: NB_EXP_40_CV_0_TFL_224_BGS_PHASE_2_COS_TTA.csv)
```

looks promising for the blend. There is still an issue that gap between LB and CV is so huge


#### EXP_50
So in this experiment, I will try to use `10` fold cv and images processed using `src/dicom_to_png_3chn_bg.py` and rescaled to size of `300`
```
MODEL:           EfficientNetB0
NUM_CLASSES:     6
BS:              380
SZ:              300
VALID:           1 FOLD CV (FOLD=0) (10 Folds)
TFMS:            get_transform()
PRETRAINED:      True (Imagenet Stats)
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=0.004, epoch=15, decay_start=0.4)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_50_CV_0_300_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.045853 	
MODEL VAL_LOSS:  0.054660	
ACCURACY THRES:  0.980348
LB SCORE:         (SUB_NAME: NB_EXP_40_CV_0_TFL_224_BGS_PHASE_2_COS.csv)
LB SCORE_TTA:     (SUB_NAME: NB_EXP_40_CV_0_TFL_224_BGS_PHASE_2_COS_TTA.csv)
```

looks promising for the blend. There is still an issue that gap between LB and CV is so huge
 ### DATA processing 
 
it seems like there is a multiple ways to process the windows:

1) ranges to normalize images: -50–150, 100–300 and 250–450. The first HU range was chosen to boost the difference between hemorrhage regions and normal tissues (supplement https://rd.springer.com/content/pdf/10.1007%2Fs00330-019-06163-2.pdf)

2) Windows used were brain window (l = 40, w = 80), bone window (l = 500, w = 3000) and subdural window (l = 175, w = 50) https://arxiv.org/pdf/1803.05854.pdf
