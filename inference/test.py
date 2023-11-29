import os
import torch
import sys
sys.path.append(".")
sys.path.append("..")
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
import time
import numpy as np
from collections import OrderedDict as odict


def main(tqdm_val, opt, model):
    if opt.latent:
        for i, data in enumerate(tqdm_val):
            model.set_input(data)
            if opt.sname != '0':
                if data['fname'][0][:-4] == opt.sname:
                    model.test()
                    res = model.get_current_visuals()
                    if opt.save_imgs:
                        folder_dir = './ckpt/%s/output' % (opt.name)  
                        os.makedirs(folder_dir, exist_ok=True)
                        for i in range(opt.latent_n*2):
                            save_dir = '%s/%s.png' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0]+'_'+str(i))
                            dataset_test.imio.write(np.array(res['data_out'][i].cpu()).astype(np.uint8), save_dir)
            else:
                model.test()
                res = model.get_current_visuals()
                
                if opt.save_imgs:
                    folder_dir = './ckpt/%s/output' % (opt.name)  
                    os.makedirs(folder_dir, exist_ok=True)
                    for i in range(opt.latent_n*2):
                        save_dir = '%s/%s.png' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0]+'_'+str(i))
                        dataset_test.imio.write(np.array(res['data_out'][i].cpu()).astype(np.uint8), save_dir)
    else:
        psnr = 0
        time_val = 0
        for i, data in enumerate(tqdm_val):
            torch.cuda.empty_cache()
            model.set_input(data)
            torch.cuda.synchronize()
            time_val_start = time.time()
            model.test()
            torch.cuda.synchronize()
            time_val += time.time() - time_val_start
            res = model.get_current_visuals()
            if opt.save_imgs:
                folder_dir = './ckpt/%s/output' % (opt.name)  
                os.makedirs(folder_dir, exist_ok=True)
                save_dir = '%s/%s.png' % (folder_dir, os.path.basename(data['fname'][0]).split('.')[0])
                dataset_test.imio.write(np.array(res['data_out'][0].cpu()).astype(np.uint8), save_dir)

        print('Time: %.3f s AVG Time: %.3f ms' % (time_val, time_val/dataset_size_test*1000))


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.inference = True

    load_iter = opt.load_iter
    dataset_names = opt.dataset_name

    # real-world data is devided in train and validation only. 

    if opt.datatype == 'real':
        dataset = create_dataset(dataset_names, 'val', opt)
        datasets = tqdm(dataset)
    else:
        dataset = create_dataset(dataset_names, 'test', opt)
        datasets = tqdm(dataset)


    opt.load_iter = load_iter
    model_old = torch.load(opt.pre_path)
    model = create_model(opt)
    model.setup(opt)


    if len(opt.gpu_ids) == 0:
        new_state_dict = odict()
        for k, v in model_old.items():
            name = k.split('.',1)[1]
            new_state_dict[name] = v
        model.netMetaISPNet.load_state_dict(new_state_dict)
    else:
        model.netMetaISPNet.load_state_dict(model_old)

    model.eval()

    opt.dataset_name = dataset_names
    tqdm_val = datasets
    dataset_test = tqdm_val.iterable
    dataset_size_test = len(dataset_test)
    tqdm_val.reset()

    main(tqdm_val, opt, model)

    


