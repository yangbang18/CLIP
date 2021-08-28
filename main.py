import torch
import clip
import os
from misc import Constants
import argparse
import yaml
from misc.visual_memory import (
    add_visual_memory_specific_args,
    get_preliminary, 
    generate_visual_memory, 
    plot_visual_memory_example,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'Youtube2Text'])
    parser.add_argument('-root', '--root', type=str, default='/home/yangbang/new_VC_data')
    parser.add_argument('-sp', '--save_path', type=str, default='')
    parser.add_argument('-fp', '--all_frames_path', type=str, default='')
    parser.add_argument('-hpp', '--hparams_path', type=str, required=True)
    parser.add_argument('-nf', '--n_frames', type=int, default=8)
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-bsz', '--batch_size', default=64, type=int)
    parser = add_visual_memory_specific_args(parser)
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)
    
    if not args.all_frames_path:
        args.all_frames_path = os.path.join(args.root, args.dataset, 'all_frames')
    assert os.path.exists(args.all_frames_path)
    
    assert os.path.exists(args.hparams_path)
    hparams = yaml.load(open(args.hparams_path))
    args.opt = hparams['opt']

    if not args.save_path:
        args.save_path = os.path.join(args.root, args.dataset, 'visual_memory_CLIP')
        os.makedirs(args.save_path, exist_ok=True)

    # start running
    wid2relevant, file_field, vocab, encoded_image_feats = get_preliminary(args, model, preprocess, device)
    
    if args.visual_memory_example_word:
        print('- Showing {} most relevant visual content for the specified word `{}`'.format(
            args.visual_memory_example_topk, args.visual_memory_example_word))
        
        path_to_load_videos = args.path_to_load_videos
        if not path_to_load_videos:
            path_to_load_videos = os.path.join(args.root, args.dataset, 'all_videos')
            
        print('- The path to load video files is {}'.format(path_to_load_videos))
        if not os.path.exists(path_to_load_videos):
            raise FileNotFoundError('Please pass the argument `--path_to_load_videos $path` to specify the path to load video files')

        os.makedirs(args.visual_memory_example_save_path, exist_ok=True)
        save_name = args.visual_memory_example_save_name
        if not save_name:
            save_name = '{}.png'.format(args.visual_memory_example_word)
        save_path = os.path.join(args.visual_memory_example_save_path, save_name)

        plot_visual_memory_example(
            wid2relevant, 
            word=args.visual_memory_example_word,
            topk=args.visual_memory_example_topk, 
            vocab=vocab,
            path_to_load_videos=path_to_load_videos,
            video_suffix=args.video_suffix,
            n_frames=args.n_frames,
            save_path=save_path
        )
    else:
        generate_visual_memory(args, wid2relevant, file_field, encoded_image_feats)


'''
python main.py -hpp /home/yangbang/mygit/NACF-pl/experiments/MSRVTT/EQTF/base/default/version_0/hparams.yaml \
--all_frames_path /home/yangbang/new_VC_data/MSRVTT/all_frames \
-vm_topk 5 -vm_topk_per_video 1 \
-t 1.0 \
-vm_modality mi \
-vm_source_type raw \
-nf 28 \
-arch ViT-B/32

python main.py -hpp /home/yangbang/mygit/NACF-pl/experiments/MSRVTT/EQTF/base/default/version_0/hparams.yaml \
--all_frames_path /home/yangbang/new_VC_data/MSRVTT/all_frames \
-vm_topk 5 -vm_topk_per_video 1 \
-t 1.0 \
-vm_source_type hidden \
-nf 28 \
-arch ViT-B/32

python main.py -hpp /home/yangbang/NACF-pl/experiments/MSRVTT/Transformer_in_titanx/base/default/version_0/hparams.yaml \
--all_frames_path /home/yangbang/new_VC_data/MSRVTT/all_frames \
--path_to_load_videos ~/new_VC_data/MSRVTT/all_videos \
-vme_topk 15 -vme_word stroller
'''
