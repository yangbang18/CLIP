import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from misc import Constants
import pickle

from misc.utils import(
    resampling,
    get_random_ids_from_the_whole,
    get_random_ids_from_k_snippets,
    get_uniform_ids_from_k_snippets,
)

def get_frame_ids(n_total_frames, n_frames, random_type):
    if random_type == 'all_random':
        return get_random_ids_from_the_whole(n_total_frames, n_frames)
    elif random_type == 'segment_random':
        return get_random_ids_from_k_snippets(n_total_frames, n_frames)
    elif random_type == 'equally_sampling':
        return get_uniform_ids_from_k_snippets(n_total_frames, n_frames)
    else:
        raise ValueError('We do not support `random_type` = {} now'.format(random_type))


def get_ids_set(specific, mode, split=None, split_category=None):
    if specific != -1:
        # we only evaluate partial examples with a specific category (MSRVTT, [0, 19])
        ids_set = [int(item) for item in split_category[mode][specific]]
    else:
        # we evaluate all examples regardless of categories
        ids_set = [int(item) for item in split[mode]]
    return ids_set


class VideoOnlyDataset(Dataset):
    def __init__(self, opt, mode, random_type, specific=-1, **kwargs) -> None:
        '''
            the argument `opt` must have the following keys:
                - info_corpus:      str, path to load the preprocessed corpus file
                - modality:         str, specify which modalities to use
                - feats_x:          str or list of str, path(s) to load features of the modality `x`
                - dim_x:            int, the dimension of the features of the modality `x`
                - n_frames:         int, the number of sampled frames for each video
                - load_feats_type:  int, choices = [0, 1] 
        '''
        Dataset.__init__(self)
        assert mode in ['train', 'validate', 'test']
        assert random_type in ['segment_random', 'all_random', 'equally_sampling']
        self.opt = opt
        self.mode = mode
        self.random_type = random_type
        
        info = pickle.load(open(opt['info_corpus'], 'rb'))['info']
        self.itoc = info.get('itoc', None)

        self.ids_set = get_ids_set(
            specific=specific, 
            mode=mode, 
            split=info['split'], 
            split_category=info.get('split_category', None),
        )

    def __getitem__(self, index):
        vid = 'video%d' % self.ids_set[index]
        return self._getitem_video_only(vid)

    def __len__(self):
        return len(self.ids_set)
    
    def _getitem_video_only(self, vid):
        if not hasattr(self, 'databases'):
            self.databases = self._make_databases()
        
        return self.get_video_features_by_vid(vid)
    
    def _make_databases(self):
        def _load_database(path):
            if not path: return []
            if not isinstance(path, list): path = [path]
            return [h5py.File(p, 'r') for p in path if '.hdf5' in p]

        databases = []
        for char in self.opt['modality'].lower():
            key_name = "feats_%s" % char
            database = _load_database(self.opt[key_name])
            assert len(database) > 0
            databases.append([key_name, database, self.opt["dim_%s" % char]])
        return databases

    def get_video_features_by_vid(self, vid):
        _dict = {'video_ids': vid}
        
        frame_ids = get_frame_ids(
            self.opt.get('n_total_frames', 60),
            self.opt['n_frames'], 
            self.random_type
        ) if self.opt['load_feats_type'] == 0 else None

        if frame_ids is not None:
            _dict['frame_ids'] = frame_ids

        _dict['feats'] = []
        for item in self.databases:
            feats = self._load_feats(item[1:], vid, frame_ids=frame_ids)
            _dict['feats'].append(torch.FloatTensor(feats))

        if self.itoc is not None:
            _dict['category'] = torch.LongTensor([self.itoc[int(vid[5:])]])

        return _dict
    
    def _load_feats(self, data, vid, **kwargs):
        frame_ids = kwargs.get('frame_ids', None)
        padding = kwargs.get('padding', True)

        databases, dim = data
        max_seq_len = databases[0].get('max_len', self.opt['n_frames'])
        if max_seq_len != self.opt['n_frames']:
            max_seq_len = int(np.asarray(max_seq_len))

        feats = []
        pre_len = None
        for database in databases:
            if vid not in database.keys():
                if padding:
                    return np.zeros((max_seq_len, dim))
                else:
                    return np.zeros(dim)
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1 and padding:
                    if pre_len is not None:
                        data = data[np.newaxis, :].repeat(pre_len, axis=0)
                    else:
                        data = data[np.newaxis, :].repeat(self.opt.get('n_total_frames', 60), axis=0)
                else:
                    pre_len = data.shape[0]
            feats.append(data)

        if len(feats[0].shape) == 1:
            feats = np.concatenate(feats, axis=0)
            return feats

        feats = np.concatenate(feats, axis=1)

        if self.opt['load_feats_type'] == 0:
            assert frame_ids is not None
        elif self.opt['load_feats_type'] == 1:
            source_length = feats.shape[0]
            if source_length >= self.opt['n_frames']:
                frame_ids = get_frame_ids(
                        source_length, 
                        self.opt['n_frames'], 
                        self.random_type)
            else:
                frame_ids = resampling(source_length, max_seq_len)
        else:
            source_length = feats.shape[0]
            if source_length < max_seq_len:
                frame_ids = resampling(source_length, max_seq_len)
            else:
                frame_ids = [_ for _ in range(feats.shape[0])]

        return feats[frame_ids]


class TextOnlyDataset(Dataset):
    def __init__(self, opt, mode, n_caps_per_video, specific=-1, make_infoset=True, **kwargs) -> None:
        '''
            the argument `opt` must have the following keys:
            * simplest version:
                - info_corpus:      str, path to load the preprocessed corpus
                - max_len:          int, maximun length of captions
            * full version:
                - info_corpus:      str, path to load the preprocessed corpus
                - references:       str, path to load references (ground-truth captions w/o preprocessing)
                - max_len:          int, maximun length of captions
                - seed:             int, random seed
                - decoding_type:    str, choices = ['ARFormer', 'NARFormer']
                - visual_word_generation: bool, whether preparing data for the auxiliary task or not
                - beta:             list of two float numbers, specify the lowest and highest masking prob for MLM
                - demand:           list of str, specify which types of words will be treated as visual words,
                                    e.g., ['NOUN', 'VERB']
        '''
        Dataset.__init__(self)
        assert mode in ['train', 'validate', 'test']
        assert n_caps_per_video >= 0
        self.opt = opt
        self.mode = mode
        self.n_caps_per_video = n_caps_per_video

        data = pickle.load(open(opt['info_corpus'], 'rb'))
        self.captions = data['captions']
        self.pos_tags = data['pos_tags']

        info = data['info']    
        self.itow = info['itow']
        self.itoc = info.get('itoc', None)        
        self.itop = info.get('itop', None)
        self.length_info = info.get('length_info', None)
        self.vid2attr = info.get('vid2attr', None)
        self.random = np.random.RandomState(opt.get('seed', 0))
        
        self.ids_set = get_ids_set(
            specific=specific, 
            mode=mode, 
            split=info['split'], 
            split_category=info.get('split_category', None),
        )

        if make_infoset:
            self.infoset = self._make_infoset()
    
    def __getitem__(self, index):
        return self._getitem_text_only(index)

    def _getitem_text_only(self, index):
        vid = self.infoset[index]['vid']
        cap_id = self.infoset[index]['cap_id']
        labels = self.infoset[index]['labels']
        taggings = self.infoset[index]['pos_tags']

        data = {'video_ids': vid}
        data.update(self._prepare_input_ids(cap_id, labels, taggings))

        data['category'] = torch.LongTensor([self.infoset[index]['category']])
        data['length_target'] = torch.FloatTensor(self.infoset[index]['length_target'])
        data['tgt_visual_taggings'] = torch.LongTensor(self._prepare_tgt_visual_taggings(labels, taggings))
        if self.vid2attr is not None:
            data['labels_attr'] = torch.FloatTensor(self.vid2attr[vid])
        return data
        
    def __len__(self):
        return len(self.infoset)
    
    def _make_infoset(self):
        infoset = []

        for idx in self.ids_set:
            vid = 'video%d' % idx
            category = self.itoc[idx] if self.itoc is not None else 0
            captions = self.captions[vid]
            pos_tags = self.pos_tags[vid] if self.pos_tags is not None else ([None] * len(captions))
            assert len(captions) == len(pos_tags)

            # prepare length info for each video example, only if decoding_type == 'NARFormmer'
            # e.g., 'video1': [0, 0, 3, 5, 0]
            if self.length_info is None or vid not in self.length_info.keys():
                length_target = np.zeros(self.opt['max_len'])
            else:
                length_target = self.length_info[vid]
                length_target = length_target[:self.opt['max_len']]
                if len(length_target) < self.opt['max_len']:
                    length_target += [0] * (self.opt['max_len'] - len(length_target))

                length_target = np.array(length_target) / sum(length_target)
            
            # decide which captions are used to calculate training/evaluation loss
            if self.n_caps_per_video == 0:
                cap_id_set = [i for i in range(len(captions))]
            elif self.n_caps_per_video == 1 and self.mode != 'train':
                cap_id_set = [0]
            else:
                n_caps_per_video = min(len(captions), self.n_caps_per_video)
                cap_id_set = self.random.choice(
                    [i for i in range(len(captions))], 
                    n_caps_per_video,
                    replace=False
                )
            
            for cap_id in cap_id_set:
                item = {
                    'vid': vid,
                    'labels': captions[cap_id],
                    'pos_tags': pos_tags[cap_id],
                    'category': category,
                    'length_target': length_target,
                    'cap_id': cap_id,
                    }
                infoset.append(item)

        if hasattr(self, '_make_infoset_post_processing'):
            infoset = self._make_infoset_post_processing(infoset)
        
        return infoset

    def _prepare_tgt_visual_taggings(self, labels, pos_tagging):
        """
        Get visual tagging from pos_tagging and target sentence.
        Because we want to remove be words from visual words,
        and VERB from pos_tag includes be words, we could not directly use pos_tag instead.
         example:
          sentence        "<bos> a man is watching movie on his phone <eos>"
          visual   tag    [  0   0  1   0    1       1    0  0    1     0 ] with padding
        Notice that <bos> should be remove to match label!
        """
        # remember to exclude <bos> <eos>
        assert self.itop and self.itow

        # sentence is
        # " ".join([self.itow[l] for l in labels])

        # get the position of tokens that have the pos_tag we demand
        visual_word_tag = [0]  # 0 for <bos>
        for i, item in enumerate(pos_tagging[1:-1]):
            w = self.itow[labels[i+1]]
            # we ignore verb ``be''
            if self.itop[item] in ['VERB', 'NOUN'] and w not in ['is', 'are', 'was', 'were', 'be']:
                visual_word_tag.append(1)
            else:
                visual_word_tag.append(0)
        return self._padding(visual_word_tag, add_eos=True)[1:]

    def _prepare_input_ids(self, cap_id, labels, taggings):
        _dict = {'caption_ids': cap_id}

        results = self._make_source_target(labels, taggings)
        tokens, labels, taggings = map(
            lambda x: results[x], 
            ["dec_source", "dec_target", "tagging"]
        )
        tokens_1 = results.get('dec_source_1', None)
        labels_1 = results.get('dec_target_1', None)

        _dict['input_ids'] = torch.LongTensor(tokens)
        _dict['labels'] = torch.LongTensor(labels)

        if taggings is not None:
            _dict['taggings'] = torch.LongTensor(taggings)
        if tokens_1 is not None:
            _dict['tokens_1'] = torch.LongTensor(tokens_1)
            _dict['labels_1'] = torch.LongTensor(labels_1)

        return _dict

    def _make_source_target(self, target, tagging):
        if self.opt.get('decoding_type', 'ARFormer') == 'NARFormer':
            results = self._source_target_mlm(target[1:-1]) # exclude <bos> <eos>
        else:
            # ARFormer
            results = {
                'dec_source': self._padding(target, add_eos=True)[:-1], 
                'dec_target': self._padding(target, add_eos=True)[1:]
            }

        assert len(results['dec_source']) == len(results['dec_target'])

        if self.opt.get('visual_word_generation', False):
            results.update(self._source_target_visual_word(target=target, pos_tag=tagging))

        if 'tagging' not in results.keys():
            results['tagging'] = self._padding(tagging, add_eos=True)

        return results

    def _source_target_mlm(self, target):
        assert target[0] != Constants.BOS
        assert target[-1] != Constants.EOS

        beta_low, beta_high = self.opt.get('beta', [0, 1])

        min_num_masks = 1
        dec_source = torch.LongTensor(target)
        dec_target_cp = torch.LongTensor(target)
        dec_target = torch.LongTensor([Constants.PAD] * len(dec_source))

        if self.mode == 'train':
            if min_num_masks >= len(dec_source):
                ind = np.array([],dtype=np.uint8)
            else:
                low = max(int(len(dec_source) * beta_low), min_num_masks)
                high = max(int(len(dec_source) * beta_high), min_num_masks)
                if high == low:
                    high += 1
                sample_size = self.random.randint(low, high)
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            if len(ind):
                dec_source[ind] = Constants.MASK
                dec_target[ind] = dec_target_cp[ind]
        else:
            dec_source[dec_source!=Constants.PAD] = Constants.MASK
            dec_target = dec_target_cp           

        dec_source = self._padding(dec_source.tolist(), add_eos=False)
        dec_target = self._padding(dec_target.tolist(), add_eos=False)
        
        return {'dec_source': dec_source, 'dec_target': dec_target}

    def _source_target_visual_word(self, **kwargs):
        target = kwargs['target']
        pos_tag = kwargs['pos_tag']
        sent_length = len(target[1:-1]) # exclude <bos> <eos>

        visual_tag = Constants.VIS
        target_tag = Constants.MASK

        if self.mode != 'train':
            dec_target_1 = [0]
            dec_source_1 = [0]
        else:
            assert len(target) == len(pos_tag)
            assert self.itop is not None

            dec_source_1 = self._padding(
                [visual_tag] * (sent_length if self.opt['decoding_type'] == 'NARFormer' else len(target)), 
                add_eos=False if self.opt['decoding_type'] == 'NARFormer' else True
            )

            # get the position of tokens that have the pos_tag we demand
            pos_satisfied_ind = []
            for i, item in enumerate(pos_tag[1:-1]):
                w = self.itow[target[i+1]]
                # we ignore verb ``be''
                if self.itop[item] in self.opt['demand'] and w not in ['is', 'are', 'was', 'were', 'be']:
                    pos_satisfied_ind.append(i)

            pos_satisfied_ind = np.array(pos_satisfied_ind)
            
            # decoder1 need to predict tokens with satisfied pos_tag from scratch
            # meanwhile, decoder1 should learn to keep the remaining tokens (i.e., <mask>) unchanged
            dec_target_1 = torch.LongTensor([target_tag] * sent_length)
            dec_target_cp = torch.LongTensor(target[1:-1])
            dec_target_1[pos_satisfied_ind] = dec_target_cp[pos_satisfied_ind]

            if self.opt['decoding_type'] == 'NARFormer':
                dec_target_1 = self._padding(dec_target_1.tolist(), add_eos=False)
            else:
                # when training with autoregressive transformer, the first token will be ignored, i.e., label = dec_target_1[1:]
                dec_target_1 = self._padding([target[0]] + dec_target_1.tolist() + [Constants.EOS], add_eos=True)

        return {'dec_source_1': dec_source_1, 'dec_target_1': dec_target_1}

    def _padding(self, seq, add_eos=True):
        if seq is None:
            return None
        res = seq.copy()
        if len(res) > self.opt['max_len']:
            res = res[:self.opt['max_len']]
            if add_eos:
                res[-1] = Constants.EOS
        else:
            res += [Constants.PAD] * (self.opt['max_len'] - len(res))
        return res
    
    def get_references(self):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return self.references

    def get_preprocessed_references(self):
        return self.captions

    def get_gt_sentences_by_vid(self, vid):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return [item['caption'] for item in self.references[vid]]

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.itow


class JointDataset(VideoOnlyDataset, TextOnlyDataset):
    def __init__(self, opt, mode, print_info=False, specific=-1, **kwargs):
        if mode != 'train' or kwargs.get('is_validation', False):
            random_type = 'equally_sampling'
            n_caps_per_video = 1 if not kwargs.get('all_caps', False) else 0
        else:
            random_type = opt.get('random_type', 'segment_random')
            n_caps_per_video = opt.get('n_caps_per_video', 0)

        VideoOnlyDataset.__init__(self, opt, mode, random_type, specific, **kwargs)
        TextOnlyDataset.__init__(self, opt, mode, n_caps_per_video, specific, **kwargs)
        
        if print_info:
            self.print_info()
    
    def print_info(self):
        print('Dataset Information:')
        print('- the number of videos in the set `{}`: {}'.format(
            self.mode, len(self.ids_set))
        )
        print('- the number of samples (n_caps_per_video={}): {}'.format(
            self.n_caps_per_video, len(self.infoset))
        )
        print('- vocab size is', len(self.itow))
        print('- the maximum sequence length (max_len) is set to', self.opt['max_len'])
        
        print('Modality Information:')
        for char in self.opt['modality'].lower():
            print('- loading feats_{} ({}) from {}'.format(
                char, self.opt['dim_' + char], self.opt['feats_' + char])
            )
        print('- load feats type: %d' % self.opt['load_feats_type'])
        print('- the number of sampled frames is set to', self.opt['n_frames'])

    def get_specific_data_by_vid_and_cap_id(self, vid, cap_id, device='cpu'):
        data = self.get_video_features_by_vid(vid)

        label = self.captions[vid][cap_id]
        tagging = self.pos_tags[vid][cap_id]
        data.update(self._prepare_input_ids(cap_id, label, tagging))

        for k in data.keys():
            if k not in ['frame_ids', 'video_ids', 'caption_ids']:
                if isinstance(data[k], list):
                    data[k] = [item.unsqueeze(0).to(device) for item in data[k]]
                else:
                    data[k] = data[k].unsqueeze(0)
                    data[k] = data[k].to(device)

        return data

    def _make_infoset_post_processing(self, infoset):
        return infoset

    def __getitem__(self, index):
        vid = self.infoset[index]['vid']
        
        data = {}
        data.update(self._getitem_video_only(vid))
        data.update(self._getitem_text_only(index))
        
        if 'rnn' in self.opt.get('decoder', '').lower():
            # if a video has the category of 1, 
            # then [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...] is treated as category features
            category_one_hot = [0] * self.opt.get('num_category', 20)
            category_one_hot[self.infoset[index]['category']] = 1
            data['category'] = torch.FloatTensor(category_one_hot)

        return data

    def __len__(self):
        return len(self.infoset)


def get_loader(opt, mode, print_info=False, specific=-1, **kwargs):
    dataset_type = kwargs.get('dataset_type', 'joint')
    if dataset_type == 'video':
        dataset_class = VideoOnlyDataset
    elif dataset_type == 'text':
        dataset_class = TextOnlyDataset
    else:
        dataset_class = JointDataset

    dataset = dataset_class(opt, mode, print_info=print_info, specific=specific, **kwargs)
    batch_size = kwargs.get('batch_size', opt.get('batch_size', 64))
    
    if kwargs.get('all_samples_one_batch', False):
        batch_size = len(dataset)

    not_shuffle = kwargs.get('not_shuffle', False)
    num_workers = kwargs.get('num_workers', opt.get("num_workers", 0))

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True if (mode=='train' and not not_shuffle) else False,
        num_workers=num_workers,
    )