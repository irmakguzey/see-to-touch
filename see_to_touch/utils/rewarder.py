import numpy as np
import torch
import ot
import numpy as np

from skimage.metrics import structural_similarity as ssim

def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method='sinkhorn_gpu',
                           niter=500,
                           epsilon=0.01,
                           exponential_weight_init=False):
    if exponential_weight_init:
        a = 2/3 # We will be implmenting 
        r = 1/3 # We are approximating these initial values - by looking at the plots
        N_x = X.shape[0]
        N_y = Y.shape[0]
        X_pot = [a * (r**(N_x-n)) for n in range(N_x)]
        Y_pot = [a*(r**(N_y-n)) for n in range(N_y)]
    else:
        X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
        Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = ot.sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan


def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, 2))
    return c


def structural_similarity_index(x, y, base_factor):
    # We assume that x and y are arrays of images
    print('IN SSIM - x.shape: {}, y.shape: {}'.format(
        x.shape, y.shape
    ))
    ssim_matrix = np.zeros((x.shape[0], y.shape[0]))
    for i,x_img in enumerate(x):
        for j,y_img in enumerate(y):
            if len(x_img.shape) > 2: # If the images are RGB images 
                channel_axis = 2 if x_img.shape[2] == 3 else 0
            else:
                channel_axis = None # Images are grayscale
            
            print('x_img.shape: {}, y_img.shape: {} in structural_similarity_index()'.format(
                x_img.shape, y_img.shape
            ))

            try:
                x_img = x_img.numpy()
            except:
                pass
            try:
                y_img = y_img.numpy() 
            except:
                pass

            ssim_value = ssim(
                x_img,
                y_img,
                data_range = max(x_img.max(), y_img.max()) - min(x_img.min(), y_img.min()),
                channel_axis = channel_axis
            )
            # Normalize the ssim value to have a distinctive reward 
            ssim_value = (ssim_value - base_factor) / (1 - base_factor)
            ssim_matrix[i,j] = ssim_value 

    return ssim_matrix



