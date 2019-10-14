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
| EXP_20                | Res2Net50      | False            |                 | 512| 40/80                 | 0.979002 |0.076  | 0.074 | Did trasnfer learning from Above EXP_20, trained on 224 |
| EXP_30                | Resnext50      | True             |                 | 224| 40/80                 | 0.980641 |0.095  | 0.079 | added cutout, zoom_rand=1.4 |
| EXP_40                | xresnet50      | True             | Attention       | 224| 40/80, 80/200, 200/450                 | 0.980348 |0.083  | 0.074 |  3 channel diffrent windows, background substractued, trained using `EXP_10_MIXUP` weights|
| EXP_50                | EfficientNetB3  | True | weighted loss  | 300| 40/80, 80/200, 200/450  | 0.979881 |0.076| 0.071 |  ||
| EXP_60    | Res2Net50      | True           |   | 224| 40/80, 50/175, 500/3000  | 0.980367 |0.082  | 0.072 | |
| EXP_70    | xresnet50      | False           |   | 300| 40/80, 50/175, 500/3000  |  |  |  | |
| EXP_80    | xresnet50      | False           | weighted loss   | 224| 40/80, 80/200, 200/450   | 0.979646 |0.079|0.081||
| EXP_90    | EfficientNetB0  | True           |    | 224| 40/80, 80/200, 200/450   | 0.981537 |0.094|0.075||
| EXP_310    | EfficientNetB4 | True           |    | 380| 40/80, 80/200, 200/450   | 0.978151 |0.071|0.072||

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

 MODEL WEIGHTS:   [NB_EXP_20_CV_0_PHASE_1_COS.pth]
 MODEL TRN_LOSS:  0.053704
 MODEL VAL_LOSS:  0.059312
 ACCURACY THRES:  0.978431
 LB SCORE:        0.084 (SUB_NAME: NB_EXP_20_CV_0_COS.csv)
 LB SCORE_TTA:    0.079 (SUB_NAME: NB_EXP_20_CV_0_COS_TTA.csv)
 ```
 Comments: Training from scratch shows similar results
 
 
Ok lets continue with increasing the image size and doing

 ```
 MODEL:           Res2Net50
 NUM_CLASSES:     6
 BS:              64
 SZ:              512
 VALID:           1 FOLD CV (FOLD=0)
 TFMS:            get_transform()
 PRETRAINED:      True (NB_EXP_20_CV_0_PHASE_1_COS.pth)
 NORMALIZE:       Data

 TRAINING:        OPT: Radam
                  Policy: Cosine Anneal 
                  flattenAnneal(lr=1e-2/8, epoch=10, decay_start=0.7)-Unfrozen

 MODEL WEIGHTS:   [NB_EXP_20_CV_0_PHASE_2_COS.pth]
 MODEL TRN_LOSS:  0.051353
 MODEL VAL_LOSS:  0.057057
 ACCURACY THRES:  0.979002
 LB SCORE:        0.076 (SUB_NAME: NB_EXP_20_CV_0_2_COS.csv)
 LB SCORE_TTA:    0.074 (SUB_NAME: NB_EXP_20_CV_0_2_COS_TTA.csv)
 ```


Higher Image resolution yields good results


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
 
 
 ### DATA processing -1
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
So in this experiment, I will try to use `10` fold cv and images processed using `src/dicom_to_png_3chn_bg.py` and rescaled to size of `300`, Also for the loss I have used weights with loss `[2, 1, 1, 1, 1, 1 ]` (2 for class ANY, and once for every other )
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
MODEL TRN_LOSS:  0.054744 	
MODEL VAL_LOSS:  0.060970
ACCURACY THRES:  0.979881
LB SCORE:        0.076 (SUB_NAME: NB_EXP_50_CV_0_300_PHASE_1_COS.csv.csv)
LB SCORE_TTA:     (SUB_NAME: NB_EXP_50_CV_0_300_PHASE_1_COS_TTA.csv.csv)
```

looks promising for the blend. There is still an issue that gap between LB and CV is so huge



#### EXP_80
Using same data processing as for `src/dicom_to_png_3chn_bg.py`. But this time i am using weights. My mistake was that i used `pos_weights` in Pytorch BCE, but instead I should use just `weights` and pass `weights = torch.FloatTensor([2, 1, 1, 1, 1, 1]).cuda()`. I am traying `xresnet50`.

```
MODEL:           xresnet50
NUM_CLASSES:     6
BS:              480
SZ:              224
VALID:           1 FOLD CV (FOLD=6) (10 Fold)
TFMS:            get_transforms(max_rotate=180, flip_vert=True, max_zoom=1.4)
PRETRAINED:      False 
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=0.002, epoch=50, decay_start=0.7)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_80_CV_0_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.054669 	
MODEL VAL_LOSS:  0.075873	
ACCURACY THRES:  0.979646
LB SCORE:        0.799 (SUB_NAME: NB_EXP_80_CV_0_224_PHASE_2_COS.csv)
LB SCORE_TTA:    0.081 (SUB_NAME: NB_EXP_80_CV_0_224_PHASE_2_COS_TTA.csv)
```


#### EXP_90
Using same data processing as for `src/dicom_to_png_3chn_bg.py`. 

```
MODEL:           EfficientNet-B0
NUM_CLASSES:     6
BS:              512
SZ:              224
VALID:           1 FOLD CV (FOLD=6) (10 Fold)
TFMS:            get_transforms(max_rotate=180,
                                flip_vert=True,
                                max_zoom=1.4, 
                                max_lighting=0.3, 
                                xtra_tfms=cutout(n_holes=(10, 50), length=(9, 35), p=0.7)
PRETRAINED:      True 
NORMALIZE:       Imagenet

TRAINING:        OPT: Adam
                 Policy: OneCycle 
                 fit_one_cycle(lr=0.003, epoch=50, pct_)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_80_CV_0_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.075873 	
MODEL VAL_LOSS:  0.052711	 
ACCURACY THRES:  0.981537
LB SCORE:        0.093 (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS.csv)
LB SCORE_TTA:    0.075 (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS_TTA.csv)
```


 ### DATA processing 
 
it seems like there is a multiple ways to process the windows:

1) ranges to normalize images: -50–150, 100–300 and 250–450. The first HU range was chosen to boost the difference between hemorrhage regions and normal tissues (supplement https://rd.springer.com/content/pdf/10.1007%2Fs00330-019-06163-2.pdf)



### DATA processing -2
This time I am using diffrent windows to process images, indows used were brain window (l = 40, w = 80), subdural window (l = 50, w = 175), bone window (l = 500, w = 3000) https://arxiv.org/pdf/1803.05854.pdf. The script located here: `src/dicom_to_png_3chn_sasanak.py`. Also this time I did `10` fold split and using fold `6` to train.

#### EXP_60

```
MODEL:           res2net
NUM_CLASSES:     6
BS:              368
SZ:              224
VALID:           1 FOLD CV (FOLD=6) (10 Folds)
TFMS:            get_transform()
PRETRAINED:      False 
NORMALIZE:       Data Stats

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=0.004, epoch=30, decay_start=0.7)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_60_CV_6_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.046806 	
MODEL VAL_LOSS:  0.054757	
ACCURACY THRES:  0.980367
LB SCORE:        0.082 (SUB_NAME: NB_EXP_60_CV_6_224_PHASE_1_COS.csv)
LB SCORE_TTA:    0.072 (SUB_NAME: NB_EXP_60_CV_6_224_PHASE_1_COS_TTA.csv)
```

#### EXP_70
Using same data processing as for EXP_60. But this time i am using weights. My mistake was that i used `pos_weights` in Pytorch BCE, but instead I should use just `weights` and pass `weights = torch.FloatTensor([2, 1, 1, 1, 1, 1]).cuda()`. I am traying `xresnet50` with Attention. 
```
MODEL:           xresnet50+Attn
NUM_CLASSES:     6
BS:              480
SZ:              224
VALID:           1 FOLD CV (FOLD=0) (10 Fold)
TFMS:            get_transforms(max_rotate=180, flip_vert=True, max_zoom=1.4)
PRETRAINED:      False 
NORMALIZE:       Data

TRAINING:        OPT: Radam
                 Policy: Cosine Anneal 
                 flattenAnneal(lr=0.0028, epoch=50, decay_start=0.68)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_70_CV_6_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:  0.058169 	
MODEL VAL_LOSS:  0.070095
ACCURACY THRES:  0.980923
LB SCORE:         (SUB_NAME: NB_EXP_70_CV_6_224_PHASE_1_COS.csv)
LB SCORE_TTA:    0.824 (SUB_NAME: NB_EXP_70_CV_6_224_PHASE_1_COS_TTA.csv)
```
#### EXP_310
Using same data processing as for EXP_60. But this time i am using weights. My mistake was that i used `pos_weights` in Pytorch BCE, but instead I should use just `weights` and pass `weights = torch.FloatTensor([2, 1, 1, 1, 1, 1]).cuda()`. Also using eff `b4`.


data processing: `src/dicom_to_png_3chn_bg.py`. 

```
MODEL:           EfficientNet-B4
NUM_CLASSES:     6
BS:              184
SZ:              380
VALID:           1 FOLD CV (FOLD=0) (10 Fold)
TFMS:            get_transforms(max_rotate=180, flip_vert=True, max_zoom=1.3)

PRETRAINED:      True 
NORMALIZE:       Imagenet

TRAINING:        OPT: Adam
                 Policy: OneCycle 
                 fit_one_cycle(lr=1e-2/3, epoch=7,)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_310_CV_0_380_UNFRZ.pth]
MODEL TRN_LOSS:  0.065647 	
MODEL VAL_LOSS:  0.065829 
ACCURACY THRES:  0.978151
LB SCORE:        0.071 (SUB_NAME: NB_EXP_310_CV_0_380_PHASE_1_COS.csv)
LB SCORE_TTA:    0.072 (SUB_NAME: NB_EXP_310_CV_0_380_PHASE_1_COS_TTA.csv)
```
With Weight it shoule very closee correlation between cv and lb (Please use `PHASE_1` for submision)

____________________________________________________________
# OCT-8 UPDATED START
Finally after some playing around I manage to correctly implement loss function with weights in fastai. The problem was that if you apply weight and get prediction it wont be sigmoid.. so after predcition I have to take `torch.sigmoid`. Now my loss function resemeblce public score perfectly. 
```
weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
def my_loss(y_pred,y_true,weights = torch.FloatTensor(weights).cuda()):
    return F.binary_cross_entropy_with_logits(y_pred,
                                  y_true,
                                  weights)
                                  
```

after we get predictions we have to do `preds = torch.sigmoid(preds)`

##SETUP
from now I will use `5-fold StratifiedKFold` split (called: `val_idx.joblib`). and correct losss. I will devide my experiments in to 3 based on dataset and processing:

1) `EXP_200s` Simple one window (40/80) converted to 3 channel png using `src/dicom_to_png.py`
2) `EXP_300s` Make 3 channel RGB Image with 3 diffrent windo sizes `40/80`, `80/200` and very wide `200/450`. When doing this I also notice that there is also a lot of black backround. To remove black background and convert images to 3 channel from dicom files use script `src/dicom_to_png_3chn_bg.py`
3) `EXP_400s` This time I am using diffrent windows to process images, indows used were brain window (l = 40, w = 80), subdural window (l = 50, w = 175), bone window (l = 500, w = 3000) https://arxiv.org/pdf/1803.05854.pdf. The script located here: `src/dicom_to_png_3chn_sasanak.py`.


## EXP_300s
data processing script:`src/dicom_to_png.py`
#### EXP_300
Testing for 30 epoch 1 cycle policy 

```
MODEL:           EfficientNet-B0
NUM_CLASSES:     6
BS:              512
SZ:              224
VALID:           5 FOLD_SPLIT (FOLD - 0)
TFMS:            get_transforms(max_rotate=245,
                                flip_vert=True,
                                max_zoom=1.4, 
                                max_lighting=0.3)
PRETRAINED:      True 
NORMALIZE:       Imagenet

TRAINING:        OPT: Adam
                 Policy: OneCycle 
                 fit_one_cycle(lr=0.003, epoch=30, pct_start=0.2, wd=1e-2)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_80_CV_0_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:   	
MODEL VAL_LOSS:  	 
ACCURACY THRES:  
LB SCORE:         (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS.csv)
LB SCORE_TTA:     (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS_TTA.csv)
```


## EXP_400s
data processing script:`src/dicom_to_png.py`
#### EXP_400
Testing radam with cosine annel for 30 epoch

```
MODEL:           EfficientNet-B0
NUM_CLASSES:     6
BS:              512
SZ:              224
VALID:           5 FOLD_SPLIT (FOLD - 0)
TFMS:            get_transforms(max_rotate=320,
                                flip_vert=True,
                                max_zoom=1.5, 
                                max_lighting=0.3)
PRETRAINED:      True 
NORMALIZE:       Imagenet

TRAINING:        OPT: partial(Ranger, betas=(0.92,0.99), eps=1e-6)
                 Policy: CosineAneal 
                 flattenAnneal(lr=0.002, epoch=30, pct_start=0.7, wd=1e-2)-Unfrozen
           

MODEL WEIGHTS:   [NB_EXP_80_CV_0_224_PHASE_1_COS.pth]
MODEL TRN_LOSS:   	
MODEL VAL_LOSS:  	 
ACCURACY THRES:  
LB SCORE:         (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS.csv)
LB SCORE_TTA:     (SUB_NAME: NB_EXP_90_CV_0_224_PHASE_2_COS_TTA.csv)
```
