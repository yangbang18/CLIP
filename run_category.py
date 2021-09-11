import sys
import h5py
import torch
import clip
import os
from misc import Constants
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from misc.utils import get_uniform_ids_from_k_snippets


def visualize(pred_itoc, gt_itoc):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-nf', '--n_frames', type=int, default=28)
    parser.add_argument('-ft', '--fusion_type', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('-vis', '--visualize', default=False, action='store_true')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)
    args.arch = args.arch.replace('/', '-')

    root = os.path.join(Constants.base_data_path, args.dataset)
    itoc_save_path = os.path.join(root, f'CLIP_{args.arch}_nf{args.n_frames}_{args.fusion_type}_itoc.pkl')
    feats_load_path = os.path.join(root, 'feats', 'CLIP_{}.hdf5'.format(args.arch))
    assert os.path.exists(feats_load_path), \
        f'- {feats_load_path} not found! Please run `python run_feats.py -d {args.dataset} -arch {args.arch}` first.'

    info_corpus_path = os.path.join(root, 'info_corpus.pkl')
    info = pickle.load(open(info_corpus_path, 'rb'))['info']

    split = info['split']
    all_video_ids = split['train'] + split['validate'] + split['test']
    
    if args.visualize:
        assert os.path.exists(itoc_save_path)
        assert info.get('itoc', None) is not None
        pred_itoc = pickle.load(open(itoc_save_path, 'rb'))
        gt_itoc = info['itoc']
        visualize(pred_itoc, gt_itoc)
        sys.exit(0)

    # start running
    model.eval()
    model.to(device)

    print(f'- Sampling {args.n_frames} frames from each video to predict the category')
    frame_ids = get_uniform_ids_from_k_snippets(60, args.n_frames)
    category = [value.split('/')[0] for value in Constants.index2category.values()]
    tokenized_category = clip.tokenize(category).to(device)
    db = h5py.File(feats_load_path, 'r')
    new_itoc = {}

    for _id in tqdm(all_video_ids):
        vid = 'video%d' % _id
        feats_of_this_video = np.asarray(db[vid])[frame_ids, :]
        feats_of_this_video = torch.from_numpy(feats_of_this_video).float().to(device)

        with torch.no_grad():
            logits_per_image, logits_per_category = model(
                feats_of_this_video, tokenized_category, skip_encode_image=True)
            
            assert logits_per_image.shape == (args.n_frames, len(category))
            logtis_category = logits_per_image.mean(0) if args.fusion_type == 'mean' else \
                                logits_per_image.max(0)[0]
            
            _, pred_category = logtis_category.max(0)
            new_itoc[_id] = pred_category.cpu().item()

    db.close()
    pickle.dump(new_itoc, open(itoc_save_path, 'wb'))

'''
python run_category.py -d MSRVTT -arch RN50 -nf 28 -ft mean
python run_category.py -d MSRVTT -arch RN101 -nf 28 -ft mean
python run_category.py -d MSRVTT -arch RN50x4 -nf 28 -ft mean
python run_category.py -d MSRVTT -arch ViT-B/32 -nf 28 -ft mean

python run_category.py -d MSRVTT -arch RN50 -nf 28 -ft max
python run_category.py -d MSRVTT -arch RN101 -nf 28 -ft max
python run_category.py -d MSRVTT -arch RN50x4 -nf 28 -ft max
python run_category.py -d MSRVTT -arch ViT-B/32 -nf 28 -ft max
'''
