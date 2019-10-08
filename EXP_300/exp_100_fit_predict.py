%matplotlib inline  
from fastai.vision import *
from fastai.data_block import _maybe_squeeze
from fastai.callbacks import *
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump
from efficientnet_pytorch import EfficientNet


def strt_split(x, y, val_fn = 'val_idx.joblib', n_folds=5, random_seed = 42, path=Path('')):  
    
    try: 
        val_name = load(val_fn)
    except:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        val_name = [(val_idx, trn_idx) for trn_idx, val_idx in skf.split(x, y)]
        dump(val_name, val_fn)
    return val_name

def modified_label_from_df(self, cols:IntsOrStrs=1, label_cls:Callable=None, **kwargs):
    "Label `self.items` from the values in `cols` in `self.inner_df`."
    self.inner_df.labels.fillna('', inplace=True)
    labels = self.inner_df.iloc[:,df_names_to_idx(cols, self.inner_df)]
    assert labels.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
    if is_listy(cols) and len(cols) > 1 and (label_cls is None or label_cls == MultiCategoryList):
        new_kwargs,label_cls = dict(one_hot=True, classes= cols),MultiCategoryList
        kwargs = {**new_kwargs, **kwargs}
    return self._label_from_list(_maybe_squeeze(labels), label_cls=label_cls, **kwargs)


def flattenAnneal(learn:Learner, lr:float, n_epochs:int, start_pct:float, SUFFIX = 'PHASE_1_COS'):
    n = len(learn.data.train_dl)
    anneal_start = int(n*n_epochs*start_pct)
    anneal_end = int(n*n_epochs) - anneal_start
    phases = [TrainingPhase(anneal_start).schedule_hp('lr', lr),
             TrainingPhase(anneal_end).schedule_hp('lr', lr, anneal=annealing_cos)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='valid_loss', name = f'{EXP_NAME}_{SUFFIX}'))
    learn.fit(n_epochs)
    
    
    
    


def main():
    
    parser = argparse.ArgumentParser()



    parser.add_argument("-val_fn", 
                        help = 'train and validation index stored as joblib, if not passed it will create autamticly', 
                        default = 'val_idx.joblib',
                        type=str)
    
    parser.add_argument("-n_folds", 
                        help = 'number of folds to split train valid data', 
                        default = 5
                        type=int)
    
     parser.add_argument("-fold", 
                        help = 'which fold to use after splitting', 
                        default = 0
                        type=int)

     
    parser.add_argument("-path", 
                        help = 'Main path were data is stored', 
                        default = '..'
                        type=str)
    
    parser.add_argument("-bs", 
                        help = 'Batchsize', 
                        default = 512, 
                        type=int)
    
    parser.add_argument("-sz",
                        help = 'Image', 
                        default = 224, 
                        type=int)
    
    parser.add_argument("-img_folder_train", 
                        help = '', 
                        default = '..'
                        type=str)
    
    
  
    
    



    
    args = parser.parse_args()

    PATH =          Path(args.path)
    BS =            args.bs
    SZ =            args.sz
    FOLD =          args.fold 
    
    EXP_NAME =      f'NB_EXP_110_CV_{FOLD}_{SZ}'
    IMG_TRAIN_224 = PATH/'train_images__3chn_bg_224'
    IMG_TEST_224  = PATH/'test_images__3chn_bg_224'
    DF_TRAIN =      pd.read_csv(PATH/'train_labels_as_strings.csv')
    DF_SUBMI =      pd.read_csv(PATH/'stage_1_sample_submission.csv')

    
    
    
    try:
        os.mkdir(args.output_dir)
    except:
        pass
    
    convert_to_png_ = partial(convert_to_png,  dir_img=args.output_dir)