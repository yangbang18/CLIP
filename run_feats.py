import torch
import clip
import os
from misc import Constants
import argparse
import pickle
import h5py
from misc.visual_memory import prepare_encoded_image_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'Youtube2Text'])
    parser.add_argument('-fp', '--all_frames_path', type=str, default='')
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)
    args.arch = args.arch.replace('/', '-')
    
    if not args.all_frames_path:
        args.all_frames_path = os.path.join(Constants.base_data_path, args.dataset, 'all_frames')
    assert os.path.exists(args.all_frames_path)
    
    root = os.path.join(Constants.base_data_path, args.dataset)
    
    info_corpus = os.path.join(root, 'info_corpus.pkl')
    split = pickle.load(open(args.opt['info_corpus'], 'rb'))['info']['split']
    all_video_ids = split['train'] + split['validate'] + split['test']

    feats_save_path = os.path.join(root, 'feats')
    os.makedirs(feats_save_path, exist_ok=True)
    feats_save_path = os.path.join(feats_save_path, 'CLIP_{}.hdf5'.format(args.arch))
    
    print('- Save all feats to {}'.format(args.feats_save_path))
    db = h5py.File(feats_save_path, 'a')
    prepare_encoded_image_feats(args, model, preprocess, device, video_ids=all_video_ids, db=db)
    db.close()

'''
python run_feats.py --dataset MSRVTT --arch RN50
python run_feats.py --dataset MSRVTT --arch RN101
python run_feats.py --dataset MSRVTT --arch RN50x4
python run_feats.py --dataset MSRVTT --arch ViT-B/32
'''
