import clip
import os
from PIL.Image import Image
import numpy as np
import pickle
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from clip.model import CLIP
from dataloader import get_loader
from misc import Constants
from misc.utils import get_ids_of_keyframes


def add_visual_memory_specific_args(parent_parser: object) -> object:
    parser = parent_parser.add_argument_group(title='Settings of generating visual memory')
    parser.add_argument('-vm_topk_max', '--visual_memory_topk_max', type=int, default=500, 
            help='the maximun number of relevant visual content for words when generating vid2relevant '
            '(preliminary of visual memory generation, avoid repeative genration)')
    
    parser.add_argument('-vm_topk_per_video', '--visual_memory_topk_per_video', type=int, default=1, 
            help='the number of unique relevant visual content per video when generating vid2relevant')
    
    parser.add_argument('-vm_use_scores', '--visual_memory_use_scores', default=False, action='store_true',
            help='use attention scores rather than attention probs when generating vid2relevant')

    parser.add_argument('-vm_topk', '--visual_memory_topk', type=int, default=10, 
            help='the number of relevant visual content for words when generating memory '
            '(based on the pre-generated vid2relevant, '
            '`memory_topk` should not be larger than `visual_memory_topk_max`)')
    
    parser.add_argument('-vm_modality', '--visual_memory_modality', type=str, default='mi')
    parser.add_argument('-scale_factor', '--scale_factor', type=float, default=1.0)

    parser = parent_parser.add_argument_group(title='Settings to plot visual memory examples')
    parser.add_argument('-vme_word', '--visual_memory_example_word', type=str, default='', 
            help='show the most relevant visual content to the specific word (default to None); '
            'if specifying a word, other functions will not be called (e.g., generating memory)')
    
    parser.add_argument('-vme_topk', '--visual_memory_example_topk', type=int, default=5, 
            help='the number of the most relevant visual content to the specific word (default to 5)')
    
    parser.add_argument('-vme_save_path', '--visual_memory_example_save_path', type=str, default='visualization/visual_memory')
    parser.add_argument('-vme_save_name', '--visual_memory_example_save_name', type=str, default='')

    parser.add_argument('--path_to_load_videos', type=str, default='',
            help='The path to video files when visualizing an example; '
            'by default, it will be set to os.path.join(Constant.base_data.path, dataset, all_videos)')
    
    parser.add_argument('--video_suffix', type=str, default='mp4')
    return parent_parser


def get_wid2vids(
        loader: DataLoader
    ) -> Tuple[Dict[int, List[int]], List[int], np.ndarray]:

    wid2vids = defaultdict(set)

    for batch in tqdm(loader):
        labels = batch['labels'].cpu().numpy()

        for i in range(labels.shape[0]):
            for wid in labels[i]:
                wid2vids[wid].add(batch['video_ids'][i])
                
    return {k: list(v) for k, v in wid2vids.items()}


def generate_wid2relevant(
        args: object,
        wid2vids: Dict[str, List[str]],
        model: CLIP,
        preprocess: function,
        device: torch.device,
        vocab: Dict[int, str]
    ) -> Dict[str, np.ndarray]:
    model.eval()
    model.to(device)
    wid2relevant = {}

    for wid, vids in wid2vids.items():
        text = clip.tokenize([vocab[wid]]).to(device)
        
        images_of_this_wid = []
        for vid in vids:
            frames_path_of_this_vid = os.path.join(args.all_frames_path, vid)
            frames_ids = get_ids_of_keyframes(
                total_frames_of_a_video=len(os.listdir(frames_path_of_this_vid)),
                k=args.n_frames,
                identical=True,
                offset=1 # the first sampled frame is vid_00001.png (start from 1 rather than 0)
            )
            images_of_this_vid = []
            for idx in frames_ids:
                image_fn = 'image_{}.jpg'.format(idx)
                image_path = os.path.join(frames_path_of_this_vid, image_fn)
                images_of_this_vid.append(preprocess(Image.open(image_path)))
            
            images_of_this_vid = torch.stack(images_of_this_vid, dim=0)
            images_of_this_wid.append(images_of_this_vid)
            
        images_of_this_wid = torch.cat(images_of_this_wid, dim=0).to(device) # [n_vids * n_frames, *rest]

        with torch.no_grad():
            logits_per_image, logits_per_text = model(images_of_this_wid, text)
            assert logits_per_text.shape == (1, len(vids) * args.n_frames)
        
        if args.visual_memory_use_scores:
            relevant_results = logits_per_text[0] / args.scale_factor
        else:
            relevant_results = torch.softmax(logits_per_text[0] / args.scale_factor, dim=0)

        probs, indices = relevant_results.sort(descending=True)
        valid_n_topk = min(len(vids) * args.visual_memory_topk_per_video, args.visual_memory_topk)
        vid_record = {}
        indice_record = set()
        new_probs, new_indices = [], []
        # get unique valid_n_topk pairs
        for j, (p, indice) in enumerate(zip(probs, indices)):
            index = indice // args.n_frames
            offset = indice % args.n_frames # frame_id
            vid = int(vids[index][5:]) # type of int (without prefix `video`)

            new_indice = vid * args.n_frames + offset

            vid_record[vid] = vid_record.get(vid, 0) + 1
            if vid_record[vid] <= args.visual_memory_topk_per_video:
                if new_indice in indice_record: # same vid, same frame id
                    continue
                
                indice_record.add(new_indice)
                
                new_probs.append(p)
                new_indices.append(new_indice)
                if len(new_probs) >= valid_n_topk:
                    break
        
        # print(new_indices[:10]) 
        wid2relevant[wid] = np.array([new_probs, new_indices], dtype=np.float32)
    return wid2relevant


def get_preliminary(
        args: object, 
        model: CLIP,
        preprocess: function,
        device: torch.device,
    ) -> Tuple[str, Dict[int, np.ndarray]]:

    print('- Checking wheter wid2relevant has been generated or not:')
    measure_type = 'scores' if args.visual_memory_use_scores else 'probs'
    file_field = '{}_{}pv_{}_{}'.format(
        measure_type, 
        args.visual_memory_topk_per_video, 
        args.visual_memory_modality,
        int(args.scale_factor * 10),
    )
    wid2relevant_path = os.path.join(args.save_path, 'wid2relevant_{}.pkl'.format(file_field))
    
    if os.path.exists(wid2relevant_path):
        print('- {} exists!'.format(wid2relevant_path))
        wid2relevant = pickle.load(open(wid2relevant_path, 'rb'))
    else:
        print('- {} does not exist!'.format(wid2relevant_path))
        print('- Start generating wid2relevant:')

        loader = get_loader(args.opt, mode='train', print_info=True,
            not_shuffle=True, batch_size=args.batch_size, is_validation=True, all_caps=True
        )

        print('- Step 1: preparing wid2vids')
        wid2vids = get_wid2vids(loader)

        print('- Step 2: finding most relevant {} frames/segments for each word'.format(args.visual_memory_topk_max))
        wid2relevant = generate_wid2relevant(
            args=args,
            wid2vids=wid2vids,
            model=model,
            preprocess=preprocess,
            device=device,
            vocab=loader.dataset.get_vocab(),
        )

        with open(os.path.join(args.save_path, wid2relevant_path), 'wb') as f:
            pickle.dump(wid2relevant, f)

    return wid2relevant, file_field


def plot_visual_memory_example(
        wid2relevant: Dict[int, np.ndarray], 
        word: str, 
        topk: int, 
        vocab: Dict[int, str], 
        path_to_load_videos: str, 
        video_suffix: str, 
        n_frames: int, 
        save_path: str = ''
    ) -> None:
    # packages for visualizing examples
    from pretreatment.extract_frames_from_videos import extract_frames
    from glob import glob
    from PIL import Image
    import matplotlib.pyplot as plt
    import shutil

    word2wid = {v: k for k, v in vocab.items()}
    assert word in word2wid.keys(), \
        'Sorry, the specified word `{}` can not be found in the vocabulary'.format(word)
    
    assert word not in [Constants.PAD_WORD, Constants.BOS_WORD, Constants.MASK_WORD, Constants.VIS_WORD]

    wid = word2wid[word]
    relevant = wid2relevant[wid] # [2, topk]
    valid_topk = min(topk, relevant.shape[1])
    relevant_probs, relevant_indices = relevant[:, :valid_topk]
    print(relevant_indices)

    # sample frames from relevant videos first and then load relevant images (frames)
    frames_path = './tmp_visualizing_visual_memory_of_a_word'
    images = []
    for i, indice in enumerate(relevant_indices):
        vid = 'video{}'.format(int(indice) // n_frames)
        frame_id = int(indice) % n_frames
        print('- {}: vid({}), fid({}), prob({:.4f})'.format(i, vid, frame_id, relevant_probs[i]))

        path_to_the_video = os.path.join(path_to_load_videos, '{}.{}'.format(vid, video_suffix))
        assert os.path.exists(path_to_the_video), \
            'Sorry, we can not find the video in {}'.format(path_to_the_video)
        
        # frames will be saved as vid_00001.png vid_00002.png ...
        extract_frames(
            video=path_to_the_video,
            dst=frames_path,
            prefix='{}_'.format(vid),
            suffix='png',  
            cleanup=False,
            strategy=0 # extract all frames of the video
        )

        n_total_frames = len(glob(os.path.join(frames_path, vid)))
        ids = get_ids_of_keyframes(
            total_frames_of_a_video=n_total_frames,
            k=n_frames,
            identical=True,
            offset=1 # the first sampled frame is vid_00001.png (start from 1 rather than 0)
        )

        image_name = '{}_{:05d}.png'.format(vid, ids[frame_id])
        image = Image.open(os.path.join(frames_path, image_name))
        images.append(image)

    # visualize
    fig = plt.figure(dpi=300)
    for i in range(valid_topk):
        ax = plt.subplot(1, valid_topk, i + 1)
        ax.imshow(images[i])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Prob: {:.4f}'.format(relevant_probs[i]))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # cleanup
    shutil.rmtree(frames_path)


def generate_visual_memory(
        args: object,
        wid2relevant: Dict[int, np.ndarray], 
        file_field: str,
    ) -> None:
    opt = args.opt

    loader = get_loader(opt, mode='train', print_info=True,
        not_shuffle=True, is_validation=True, all_caps=False
    )

    # prepare features of all modalities
    feats = defaultdict(list)
    modality = opt['modality']
    
    for batch in tqdm(loader):
        this_feats = batch['feats']
        for char, f in zip(modality, this_feats):
            key = 'feats_%s' % char
            feats[key].append(f)

    for k in feats.keys():
        feats[k] = torch.cat(feats[k], dim=0).cpu().contiguous()
        
    batch = feats

    fused_memory_path = os.path.join(args.save_path, 'memory_{}_top{}.npy'.format(
        file_field, args.visual_memory_topk))
    if os.path.exists(fused_memory_path):
        print('- The fused memory has been saved in {}'.format(fused_memory_path))
        return None
    
    all_memory = []
    for char in args.visual_memory_modality:
        assert char in modality
        feats = batch['feats_{}'.format(char)] # [n_training_videos, n_frames, dim]
        feats = feats.view(-1, feats.size(-1)) # [n_training_videos * n_frames, dim]

        memory = np.zeros((opt['vocab_size'], feats.size(-1)))
        
        for wid, (probs, indices) in wid2relevant.items():
            topk_probs = torch.from_numpy(probs[:args.visual_memory_topk])
            topk_indices = indices[:args.visual_memory_topk]
            topk_feats = feats[topk_indices, :]

            result = torch.matmul(topk_probs.unsqueeze(0), topk_feats) / topk_probs.sum() # [1, topk] * [topk, dim] = [1, dim]
            memory[wid, :] = result.squeeze(0).numpy()

        all_memory.append(memory)
    
    # fusion
    fused_memory = np.concatenate(all_memory, axis=-1)
    np.save(fused_memory_path, fused_memory)
    
