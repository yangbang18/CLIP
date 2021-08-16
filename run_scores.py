import torch
import clip
import os
from misc import Constants
import argparse
import pickle
from misc.visual_memory import (
    get_encoded_image_feats
)
from misc.dataloader import get_loader
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--root', type=str, default='/home/yangbang/new_VC_data')
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-fp', '--all_frames_path', type=str, default='')
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)

    if not args.all_frames_path:
        args.all_frames_path = os.path.join(args.root, args.dataset, 'all_frames')
    assert os.path.exists(args.all_frames_path)

    args.save_path = os.path.join(args.root, args.dataset, 'visual_memory_CLIP')
    os.makedirs(args.save_path, exist_ok=True)

    opt = {
        'info_corpus': os.path.join(args.root, args.dataset, 'info_corpus.pkl'),
        'max_len': 30
    }

    info_corpus = pickle.load(open(opt['info_corpus'], 'rb'))
    args.n_frames = 60
    
    encoded_image_feats = get_encoded_image_feats(
        args, model, preprocess, device, 
        video_ids=info_corpus['info']['split']['train'],
        only_n_frames=False
    )
    
    loader = get_loader(
        opt, 
        mode='train',
        not_shuffle=True, 
        batch_size=1, 
        n_caps_per_video=0,
        dataset_type='text'
    )
    
    scores_map_of_all_train_caps = np.zeros((len(loader), opt['max_len'] - 1, 60))
    vocab = info_corpus['info']['itow']

    for i, data in tqdm(enumerate(loader)):
        labels = data['labels'][0].tolist() # (max_len, )
        vid = data['video_ids'][0]
        _id = int(vid[5:])
        
        image = encoded_image_feats[_id, :, :].to(device)
        
        index_of_eos = labels.index(Constants.EOS)
        text = clip.tokenize([vocab[wid] for wid in labels[:index_of_eos]]).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text, skip_encode_image=True)
            assert logits_per_text.shape == (index_of_eos, 60)

        scores_map_of_all_train_caps[i, :index_of_eos, :] = logits_per_text.cpu().numpy()
    
    save_path = os.path.join(args.save_path, 'scores_map_of_all_train_caps.npy')
    np.save(save_path, scores_map_of_all_train_caps)

'''
python run_relevance.py --dataset MSRVTT
'''
