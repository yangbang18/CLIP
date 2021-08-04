import random
import numpy as np


def resampling(source_length, target_length):
    return [round(i * (source_length-1) / (target_length-1)) for i in range(target_length)]


def get_uniform_ids_from_k_snippets(length, k, offset=0):
    uniform_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = (bound[i] + bound[i+1]) // 2
        uniform_ids.append(idx + offset)
    return uniform_ids


def get_random_ids_from_k_snippets(length, k, offset=0):
    random_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = np.random.randint(bound[i], bound[i+1])
        random_ids.append(idx + offset)
    return random_ids


def get_random_ids_from_the_whole(length, k, offset=0):
    random_ids = random.sample([i for i in range(length)], k)
    random_ids = [i + offset for i in random_ids]
    return sorted(random_ids)


def get_uniform_items_from_k_snippets(items, k):
    uniform_ids = get_uniform_ids_from_k_snippets(len(items), k)
    return [items[idx] for idx in uniform_ids]


def get_ids_of_keyframes(total_frames_of_a_video, k, identical=True, offset=0):
    if identical:
        ''' In our implementation, we follow two steps:
            1. extract 60 features to represent a video (see the `hdf5` feature files);
            2. feed uniformly-sampled k features into the captioning model during inference.
        '''
        assert k < 60
        uniform_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, 60) # step1
        real_ids = get_uniform_items_from_k_snippets(uniform_ids, k) # step2
    else:
        ''' the real_ids is slightly different from the one above
            e.g., with total_frames_of_a_video = 198 and k = 8,
            identical = True:  real_ids = [11, 37, 60, 87, 110, 136, 159, 186]
            identical = False: real_ids = [12, 36, 61, 86, 111, 135, 160, 185]
        '''
        real_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, k)

    if offset:
        real_ids = [idx + offset for idx in real_ids]

    return real_ids
