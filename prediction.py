import time

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import sparse

from tqdm import tqdm
from simulation import setup, step

CUBE_PARAMS = {
    'side_len': 0.3,
    'start_pos': [0, 0, 1.0],
    'start_orn': p.getQuaternionFromEuler([0] * 3),
    'rgba': [1, 0, 0, 1]
}
def collect_episode(num_steps, save_path=None):

    cube_id = setup(gui=True, gravity=-10, **CUBE_PARAMS)

    v = np.random.normal(scale=[2, 2, 2], size=(3,))
    w = np.random.normal(scale=10, size=(3,))
    p.resetBaseVelocity(cube_id, linearVelocity=v, angularVelocity=w)

    imgs = []
    for i in range(num_steps):
        img = step()
        imgs.append(img)
        # time.sleep(1/240)
    imgs = np.stack(imgs)

    if save_path is not None:
        np.save(save_path, imgs)

    p.disconnect()


def get_sdr_matrix(w, num_vals):
    j = np.arange(w).reshape((1, w))
    j = np.tile(j, (num_vals, 1))
    shifts = np.arange(num_vals).reshape((num_vals, 1))
    j += shifts
    j = j.flatten()
    i = np.arange(num_vals)
    i = np.repeat(i, w)
    coords = np.stack([i, j])
    data = 1

    return sparse.COO(coords, data, shape=(num_vals, num_vals + w - 1))


def get_rgb_to_sdr_fn(w=5):
    s = get_sdr_matrix(w, 256).todense()
    sdr_dim = s.shape[-1] * 3

    # x -> (H, W, 3)
    def rgb_to_sdr(x):
        r = s[x[...,0]]
        g = s[x[...,1]]
        b = s[x[...,2]]

        sdr = np.concatenate([r, g, b], axis=-1)
        sdr = sparse.COO.from_numpy(sdr)
        return sdr

    return (rgb_to_sdr, sdr_dim)


def add_noise(ep_id=1):
    x = np.load(f'saved_episodes/{ep_id}.npy')[:,:,:,:3]

    noise = np.random.normal(size=x.shape)
    high = noise.max()
    low = noise.min()
    noise = np.uint8(255*(noise-low)/(high-low))
    mask = np.all(x==255, axis=-1, keepdims=True)
    x = np.where(mask, noise, x)

    np.save(f'saved_episodes/{ep_id}_noisy.npy', x)


def get_sdrs(w=1, ep_id=1, noisy=False):
    filename = f'saved_episodes/{ep_id}.npy' if not noisy else \
               f'saved_episodes/{ep_id}_noisy.npy'
    x = np.load(filename)[:,:,:,:3]
    rgb_to_sdr, sdr_dim = get_rgb_to_sdr_fn(w)

    sdr_list = []
    T = x.shape[0]
    for t in range(T):
        sdr = rgb_to_sdr(x[t])
        sdr_list.append(sdr)

    sdrs = sparse.stack(sdr_list)
    sparse.save_npz(f'saved_episodes/{ep_id}_sdr.npz', sdrs)
    return sdrs


def make_cons(sdrs, same_axes, diff_axes, norm_axes, num_cons=1e4, reduce_same=True):
    """
    Randomly make connections between co-activated cells

    :param sdrs: sparse.COO instance
        Sparse ndarray representing cell activations

    :param same_axes: array_like
        Axes for which connections should be made between the same index
        Ex: if same_axes=[0] then A[i,...] -> A[i,...]

        NOTE: If same_axes has more than one element, then connections will only be made if the indices for ALL axes are the same
        Ex. if same_axes=[0, 1] then
            A[0,0,..] -> A[1,1,...]
            A[0,0,..] -> A[0,1,...]
            A[0,0,..] -> A[1,0,...]
        are all invalid connections, but
            A[0,0,...] -> A[0,0,...]
        is valid

    :param diff_axes: array_like
        Axes for which connections should be made between different indices
        Ex: if diff_axes=[0] then A[i,...] -> A[j,...], where i!=j

        NOTE: If diff_axes has more than one element, then connections will only be made if the index for ANY axis is different
        Ex: if diff_axes=[0, 1] then
            A[0,0,..] -> A[1,1,...]
            A[0,0,..] -> A[0,1,...]
            A[0,0,..] -> A[1,0,...]
        are all valid connections, but
            A[0,0,...] -> A[0,0,...]
        is invalid

    :param norm_axes: array_like
        Axes for which connections should be made based on relative position not exact position
        Ex: if norm_axes=[0] then A[i,...] -> A[j,...] becomes A[0,...] -> A[j-i,...], where i<j

    :param reduce_same: optional, bool
        Whether or not to sum along the same axis. If true, the return value is a single sparse array, otherwise the return value is a list of sparse arrays

    :return:
    """

    num_cons = int(num_cons)
    any_axes = tuple(set(range(sdrs.ndim)) - set(same_axes + diff_axes))

    orig_shape = sdrs.shape
    same_dims = [orig_shape[i] for i in same_axes]
    diff_dims = [orig_shape[i] for i in diff_axes]
    any_dims = [orig_shape[i] for i in any_axes]

    axis_order = same_axes + diff_axes + any_axes
    sdrs = sdrs.transpose(axis_order)
    ordered_norm_axes = [axis_order.index(ax) for ax in norm_axes]
    ordered_shape = sdrs.shape

    flattened_shape = [np.prod(same_dims), np.prod(diff_dims), np.prod(any_dims)]
    sdrs = sdrs.reshape(flattened_shape)

    coords_list = None
    cons_list = [] # Only if reduce_same=False
    con_shape = ordered_shape[1:]*2
    I,J,K = flattened_shape

    if num_cons*I > 5e7:
        print('WARNING: Likely not enough memory to make connections')
    for i in tqdm(range(I)):
        sdr = sdrs[i]
        j, k = sdr.nonzero()

        unique, counts = np.unique(j, return_counts=True)
        if len(unique) <= 1:
            continue
        cum_counts = np.roll(np.cumsum(counts), 1)
        cum_counts[0] = 0
        c = np.random.choice(unique, (num_cons, 2), p=counts/counts.sum())
        j_choices = c[np.not_equal(c[:,0], c[:,1])] # Remove invalid connections (e.g. A -> A)

        k_range = counts[j_choices]
        k_choices = np.random.uniform(np.zeros_like(k_range), k_range).astype(int)
        choices = cum_counts[j_choices] + k_choices

        diff_indices = np.indices(diff_dims).reshape((len(diff_dims), -1)).T
        any_indices = np.indices(any_dims).reshape((len(any_dims), -1)).T

        cj = diff_indices[j[choices]]
        ck = any_indices[k[choices]]

        coords = np.concatenate([cj, ck], axis=-1).transpose((0, 2, 1))
        for axis in ordered_norm_axes:
            offset = len(same_axes)
            coords[:, axis-offset] -= np.min(coords[:,axis-offset], axis=-1, keepdims=True)
        coords = coords.transpose((0, 2, 1)).reshape((coords.shape[0], -1)).T
        coords = coords.astype(np.int64)

        if reduce_same:
            if coords_list is None:
                coords_list = coords
            else:
                coords_list = np.concatenate([coords_list, coords], axis=-1)
        else:
            data = np.array(1, dtype=np.int32)
            cons = sparse.COO(coords, data=data, shape=con_shape)
            cons_list.append(cons)

    if reduce_same:
        data = np.array(1, dtype=np.int32)
        sum_cons = sparse.COO(coords_list, data=data, shape=con_shape)
        return sum_cons

    return cons_list


def make_connections(sdrs, p_con=1e-8, con_thresh=None):
    T, H, W, sdr_dim = sdrs.shape
    w = sdr_dim//3 - 255

    connections = sparse.zeros((H, W, sdr_dim)*2, format='coo', dtype=np.uint8)

    for t in tqdm(range(T)):
        sdr = sdrs[t]

        max_cons = sdr.nnz * (sdr.nnz - 1) / 2
        num_cons = int(max_cons * p_con)
        if t == 0: print(num_cons)

        # i -> 0x(W*3*w),1x(W*3*w),...,(H-1)x(W*3*w)
        # j -> (0x(3*w),1x(3*w),...,(W-1)*(3*w))xW
        # k -> (e.g. w=3) (255,256,257,513,514,515,771,772,773)x(H*W)
        i, j, k = sdr.nonzero()
        c = 3*w*np.random.choice(sdr.nnz//(3*w), size=(num_cons, 2)) # Choose random pixels to connect
        c = c[np.not_equal(c[:,0], c[:,1])] # Remove invalid connections (e.g. A -> A)
        c = c + np.random.choice(3*w, size=c.shape) # Choose random channels in each pixel to connect

        ci = i[c]
        ci -= np.min(ci, axis=-1, keepdims=True)
        cj = j[c]
        cj -= np.min(cj, axis=-1, keepdims=True)
        ck = k[c] # Don't normalize color dimension

        coords = np.stack([ci, cj, ck]).transpose((1, 2, 0)).reshape((-1, 6)).T
        data = np.array(1, dtype=np.uint8)

        new_connections = sparse.COO(coords, data, shape=connections.shape)
        connections = connections + new_connections

    print(connections.nnz)
    print(connections.max())

    if con_thresh is None:
        filters = connections.coords.T
    else:
        filters = sparse.argwhere(connections > 1)

    print(filters.shape)

    np.save('filters.npy', filters)
    return filters


def make_bitmask(filter, H, W, sdr_dim):
    # [i1, j1, k1, i2, j2, k2] -> [[[i1, i2], [j1, j2], [k1, k2]]]
    f = filter[[0, 3, 1, 4, 2, 5]].reshape((1, 3, 2))

    maxs = np.max(f, axis=-1)[0]
    imax = maxs[0]
    jmax = maxs[1]
    H_out = H - imax
    W_out = W - jmax
    out_dim = H_out * W_out

    r = np.arange(out_dim)
    di = r // W_out
    dj = r % W_out

    d = np.stack([di, dj, np.zeros_like(di), np.arange(di.size)], axis=-1)
    d = np.tile(d[:, :, np.newaxis], (1, 1, 2))

    f = np.concatenate([f, np.zeros_like(f[:, :1, :])], axis=1)
    f = f + d
    f = f.transpose((1, 0, 2)).reshape((4, -1))

    sparse_mask = sparse.COO(f, data=True, shape=(H, W, sdr_dim, out_dim))
    return sparse_mask


def apply_filters(sdrs, filters, match_thresh=0.5, timesteps=1):
    T, H, W, sdr_dim = sdrs.shape
    if timesteps < 0:
        timesteps = T
    activations = []
    start = time.time()

    for i in tqdm(range(filters.shape[0])):
        filter = filters[i]
        mask = make_bitmask(filter, H, W, sdr_dim)

        outs = []
        for t in range(timesteps):
            sdr = sdrs[t,...,np.newaxis]

            # Count matches and normalize by max # of matches
            match = np.logical_and(sdr, mask)
            match = np.sum(match, axis=(0, 1, 2))/2
            # *** Save match here for future credit assignment ***

            # Filter out matches below threshold and combine matches from the entire image into one boolean map
            match = match > match_thresh
            out = np.any(np.logical_and(mask, match), axis=(-2, -1))
            outs.append(out)

        activations.append(sparse.stack(outs))

    stop = time.time()
    print(f'{1000*(stop-start):.1f} ms')

    return activations



if __name__ == '__main__':
    # Collect 100 consecutive RGB frames from the simulator
    # ep_path = 'saved_episodes/1.npy'
    # collect_episode(100, save_path=ep_path)
    # add_noise()

    # Convert each RGB image into an SDR
    # sdrs = get_sdrs(w=5, noisy=True)
    # sdrs = sparse.load_npz('saved_episodes/1_sdr.npz')

    # Connect random co-activated cells in different pixels
    # Normalize connections so only relative positions of the pixels matter
    # Save normalized connections as filters coordinates formatted as: (i1, j1, k1, i2, j2, k2)
    # make_connections(sdrs)
    # sum_cons = make_cons(sdrs, same_axes=(0,), diff_axes=(1,2), norm_axes=(1,2), num_cons=5e5)
    # sparse.save_npz('1_con.npz', sum_cons)

    # sum_cons = sparse.load_npz('1_con.npz')
    # filters = sum_cons.coords.T
    # filters = sparse.argwhere(sum_cons > 2)
    # np.save('filters.npy', filters)
    #
    # filters = np.load('filters.npy')
    # print(filters)
    # n = filters.shape[0]
    # print(n)

    # i_max = filters[:, 0] + filters[:, 1]
    # j_max = filters[:, 3] + filters[:, 4]
    # r = np.sqrt(i_max**2 + j_max**2)
    # indices = np.argsort(r)[:100]
    # filters = filters[indices]
    # print(filters)


    # # Turn filter coordinates into bitmasks and sweep  them across the image like a 2D convolution
    # # Resulting match arrays are [0, 1] float values that indicate the fraction of masked cells in the sdr that are active (i.e. sum(sdr * mask)/sum(mask))
    # # Match arrays are then converted to binary arrays by checking whether each value is below or above some threshold
    # # (e.g. MAX_THRESH=0.8 corresponds to checking if 80% of values match)
    # filters = np.array([[0, 0, 260, 0, 1, 260]])
    # MATCH_THRESH = 0.8
    # activations = apply_filters(sdrs, filters, match_thresh=MATCH_THRESH, timesteps=2)

    # sorted_activations = sorted(activations, key=lambda x: x[0].nnz, reverse=True)
    # for i in range(1):
    #     plt.imshow(activations[i][0].todense())
    #     plt.show()

    # sdrs2 = sparse.stack(activations)
    # sdrs2 = sparse.moveaxis(sdrs2, 0, -1)
    # print(sdrs2.shape)
    # sparse.save_npz('saved_episodes/1_sdr2.npz', sdrs2)

    # sdrs2 => (T,H,W,C)
    sdrs2 = sparse.load_npz('saved_episodes/1_sdr2.npz')
    cons2 = make_cons(sdrs2, same_axes=(3,), diff_axes=(0,), norm_axes=(0,1,2), num_cons=1e6, reduce_same=False)

    # cons2 => (C,2,H,W,2,H,W)
    # cons2 = sparse.stack(cons2)
    # sparse.save_npz('1_con2.npz', cons2)

    # cons2 = sparse.load_npz('1_con2.npz')
    # maxs = cons2.max(axis=(1,2,3,4,5,6)).todense()
    # i = np.argsort(maxs)[::-1]
    # print(list(zip(i, maxs[i])))

    # filters2 = sparse.argwhere(cons2 > 5000)
    # print(filters2)
    #
    # useful_feats = np.unique(filters2[:,0])
    #
    # filters = np.load('filters.npy')
    # acts = sdrs2.todense()
    # for i in range(acts.shape[-1]):
    #     sdr = acts[0,...,i]
    #     plt.title(filters[i])
    #     plt.imshow(sdr)
    #     plt.imsave(f'acts/{i:03d}.png', sdr)



    # for feat in useful_feats:
    #     act = sdrs2[...,feat].todense()
    #     _, ax = plt.subplots(1, 2)
    #
    #     ax[0].imshow(act[0])
    #     ax[1].imshow(act[1])
    #     plt.title(filters[feat])
    #     plt.show()




