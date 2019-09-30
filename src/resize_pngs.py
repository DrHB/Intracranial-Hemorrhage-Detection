from fastai.vision import *
import argparse



    
def main():
    parser = argparse.ArgumentParser()
 


    parser.add_argument("-path_hr_images", 
                        help = 'path to directory with high resolution images', 
                        type=str)
    
    parser.add_argument("-path_lr_images", 
                        help = 'path_to_low_resolution_images', 
                        type=str)

    parser.add_argument("-sz", 
                        help = 'size to resizing', 
                        type=int, 
                        default=224)
    
    
    args = parser.parse_args()
    
    def resize_one(fn, i, path, size):
        dest = path/fn.relative_to(Path(args.path_hr_images))
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, size, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BICUBIC)
        img.save(dest, quality=95)
    
    
    il = ImageList.from_folder(Path(args.path_hr_images))
    sets = [(Path(args.path_lr_images), args.sz)]
    
    for p,size in sets:
        if not p.exists(): 
            print(f"resizing to {size} into {p}")
            parallel(partial(resize_one, path=p, size=size), il.items)
    
if __name__ == "__main__":
    main()
