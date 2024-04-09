import jax
import jax.numpy as jnp
from src.kmeans import SphericalKMeans
from typing import Tuple

def get_ovelapping_patch_idx(h : int, w:int, patch_size: int =3):
        """  """
        xp, yp = jnp.meshgrid(jnp.arange(w-patch_size+1), jnp.arange(h-patch_size+1))
        x, y = jnp.meshgrid(jnp.arange(patch_size), jnp.arange(patch_size))
        X = x[None, None, ...] + xp[..., None, None]
        X = X.reshape(-1, patch_size, patch_size)
        Y = y[None, None, ...] + yp[..., None, None]
        Y = Y.reshape(-1, patch_size, patch_size)

        return (Y, X)

def normalize_row(X : jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
     norm_rows = jnp.linalg.norm(X, axis=1, ord=2, keepdims=True)
     return X / (norm_rows + 1e-5), norm_rows

def kappa(x : jnp.ndarray, alpha : float):
     """ gaussian function """
     return jnp.exp(alpha * (x - 1))

def linear_gaussian_pooling(map : jnp.ndarray, beta : float = 1, sampling_factor : int = 2):
    h, w, c = map.shape
    hpool = h // sampling_factor
    conv_pool = jnp.zeros((hpool, hpool, c))
    z = jnp.stack(jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij'), axis=2)
    for i in range(0, hpool):
        for j in range(0, hpool):
            pixel = jnp.array([i*sampling_factor, j*sampling_factor])
            conv_pool = conv_pool.at[i,j].set(jnp.sum(map * jnp.exp( - beta *  jnp.sum((pixel[None, None, :] - z)**2 , axis=2))[..., None], axis = (0, 1)))

    return conv_pool


class ConvKN:
    """ CKN Layer """

    def __init__(self, patch_size:int, out_channels : int, sampling_factor : int = 2,
                 n_patch_per_img_for_kmean : int = 10) -> None:
        self.patch_size = patch_size
        self.patch_idx = None
        self.patch_pad_idx = None
        self.Z = None
        self.out_channels = out_channels
        self.sampling_factor = sampling_factor
        self.spherical_kmeans = SphericalKMeans(nb_clusters=self.out_channels, max_iter=1000)
        self.n_patch_per_img_for_kmean = n_patch_per_img_for_kmean

    def extract_patches_2d(self, img : jnp.ndarray, inference : bool = False):
        if inference :
            if self.patch_pad_idx is None:
                h, w, _ = img.shape
                self.patch_pad_idx = get_ovelapping_patch_idx(h, w, self.patch_size)
            return img[self.patch_pad_idx]
        else :
            if self.patch_idx is None :
                # to do once
                h, w, _ = img.shape
                self.patch_idx = get_ovelapping_patch_idx(h, w, self.patch_size)

            return img[self.patch_idx]
    
    def train(self, key, input_maps):
        
        # extract random patches 
        batch_size, _, _, in_channels = input_maps.shape
        X = jnp.empty((0, in_channels*self.patch_size**2))
        for i in range(batch_size):
            patches = self.extract_patches_2d(input_maps[i])
            patches = patches.reshape(-1, in_channels*self.patch_size**2)
            
            # remove constant patches
            non_cst_idx = jnp.any(patches != patches[:, [0]], axis=1) # to remove
            n_patch = min(non_cst_idx.sum(), self.n_patch_per_img_for_kmean)
            # n_patch = min(patches.shape[0], self.n_patch_per_img_for_kmean)
            key, _ = jax.random.split(key)
            X = jnp.vstack((X, 
                           jax.random.choice(key, patches[non_cst_idx], shape=(n_patch,), replace=False, axis=0))
            )
            
        X = jax.random.permutation(key, X, axis=0)
        # normalize row applying kmeans
        X, _ = normalize_row(X) 
        print(f'For training kmeans, X shape {X.shape}')
        self.Z, _ = self.spherical_kmeans.fit(X, init_centroids=self.Z) # centroids are normalized
        
        # compute linear weights
        mat = kappa(self.Z.dot(self.Z.T), alpha = 1/ .5**2) # out_channels x out_channels
        D, U = jnp.linalg.eigh(mat)
        D = D.at[D<1e-6].set(1e-6)
        inv_sqrt_D = jnp.diag(D  ** (-0.5))
        self.W = U.dot(inv_sqrt_D.dot(U.T))         
        

    def __call__(self, input_maps : jnp.ndarray):
        """
        Suppose images are squarred
        :param input_maps : array of size (B, H, W, C) B is batch size, C is channel size (e.g. 3)
        """

        if self.Z is None:
             raise Warning('Filters Z have to be initialized or learned. Call .train() befor evaluating model')

        batch_size, h, w, in_channels = input_maps.shape

        for i in range(batch_size):
            # add padding, so that there is a patch for each pixel in input map
            pad = (self.patch_size - 1)
            p = pad // 2
            image = jnp.zeros((h+pad, w+pad, in_channels))
            image = image.at[p:p+h, p:p+w, :].set(input_maps[i])

            patches = self.extract_patches_2d(image, inference=True)
            patches = patches.reshape(-1, in_channels*self.patch_size**2)

            normalized_x, norm_x = normalize_row(patches)
    
            out_map = norm_x * kappa( normalized_x.dot(self.Z.T), alpha=1/(.25)).dot(self.W)
            out_map = out_map.reshape(h, w, self.out_channels)
            # linear pooling
            beta = jnp.square(jnp.sqrt(2)/self.sampling_factor) # jnp.square(1 / (h*self.sampling_factor))

            conv_pool = linear_gaussian_pooling(out_map, beta=beta,
                                                sampling_factor=self.sampling_factor)
            if i == 0:
                out_maps = conv_pool[None, ...]
            else :
                out_maps = jnp.vstack((out_maps,
                                    conv_pool[None, ...]))
                
            if jnp.isnan(out_maps).any():
                print('i', i)
                raise Warning(f'Output map has {jnp.isnan(out_maps).sum()} NaN values')
            
        return out_maps
    

class ModelCKN:
    def __init__(self, patch_sizes : list,
                 out_channels : list,
                 subsampling_factors : list,
                 n_patch_per_img_for_kmean : int = 20):
        self.n_layers = len(out_channels)
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append( ConvKN(patch_size=patch_sizes[i], 
                                       out_channels=out_channels[i],
                                       sampling_factor=subsampling_factors[i],
                                       n_patch_per_img_for_kmean = n_patch_per_img_for_kmean)
                                       )
            
    def train(self, key, input_maps, batch_size=2000):
        keys = [key]

        N = input_maps.shape[0]
        train_idx = jax.random.choice(key, jnp.arange(N), (N//batch_size, batch_size), replace=False)
        for b in range(len(train_idx)):
            input_maps_layer = input_maps[train_idx[b]]
            keys = jax.random.split(keys[-1], self.n_layers)
            print('b', b)
            for i in range(self.n_layers):
                self.layers[i].train(keys[i], input_maps_layer)
                input_maps_layer = self.layers[i](input_maps_layer)
                print(f'Layer {i}: out max {input_maps_layer.max()} out min {input_maps_layer.min()} Z max {self.layers[i].Z.max()} W max {self.layers[i].W.max()}')
                print(f'Output map shape {input_maps_layer.shape}')

    def __call__(self, input_maps):
    
        for i in range(self.n_layers):
            input_maps = self.layers[i](input_maps)

        return input_maps