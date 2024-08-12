import cv2
import numpy as np
import torch
from tqdm import tqdm
from time import time
import os
import argparse
import torch
import torch.nn.functional as F
import math
import glob
import subprocess
import torch.nn as nn

# limit torch to 8 cores (slowdown observed with more cores)
torch.set_num_threads(8)

class NeuralNetwork(nn.Module):
    ''' Neural Network class for the model predicting the PD fitted functions '''
    def __init__(self, input_size, output_size, hidden_layers, hidden_neurons, activation_func, batch_norm, dropout):
        super(NeuralNetwork, self).__init__()
        # initialize the model
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation_func = activation_func
        self.batch_norm = batch_norm
        self.dropout = dropout

        # define the layers of the model
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_neurons))
        if dropout:
            layers.append(nn.Dropout(p=dropout))

        layers.append(self.get_activation_func())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_neurons))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(self.get_activation_func())
        layers.append(nn.Linear(hidden_neurons, output_size))

        # assemble the layers
        self.model = nn.Sequential(*layers)
        return None

    def get_activation_func(self):
        # return the activation function to use in the model
        if self.activation_func == 'relu':
            return nn.ReLU()
        elif self.activation_func == 'leakyrelu':
            return nn.LeakyReLU()
        elif self.activation_func == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

    def forward(self, x):
        # pass the data in the model
        emb = self.model(x)

        # transform the obtained embedding to the desired range
        emb_exp_slopes_a = torch.tanh(emb[:,0]) # to convert range [-oo, +oo] to [-1, 1]
        emb_exp_slopes_b = torch.tanh(emb[:,1]) # to convert range [-oo, +oo] to [-1, 1]
        emb_exp_slopes_a = emb_exp_slopes_a * 11 # to convert range [-1, 1] to [-11, 11]
        emb_exp_slopes_b = emb_exp_slopes_b * 7 - 5 # to convert range [-1, 1] to [-12, 2]
        emb_exp_slopes_a = torch.exp(emb_exp_slopes_a)
        emb_exp_slopes_b = torch.exp(emb_exp_slopes_b) 

        emb_lin_slope = torch.tanh(emb[:,3]) # to convert range [-oo, +oo] to [-1, 1]
        emb_lin_slope = emb_lin_slope * 15 # to convert range [-1, 1] to [-15, 15]
        emb_lin_slope = torch.exp(emb_lin_slope) # to convert range [-15, 15] to [0, +oo]

        emb_mse_max = torch.tanh(emb[:,2]) * 7 # to convert range [-oo, +oo] to [-7, 7]
        emb_mse_max = torch.exp(emb_mse_max) # to convert range [-7, 7] to [0, 1000+]

        # concatenate the transformed embedding
        emb_exp_slopes = torch.cat((emb_exp_slopes_a.unsqueeze(1), emb_exp_slopes_b.unsqueeze(1)), dim=1)
        emb = torch.cat((emb_exp_slopes, emb_mse_max.unsqueeze(1), emb_lin_slope.unsqueeze(1)), dim=1)
        return emb




# define the WPSNR kernel
kernel = [[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]
kernel = np.array(kernel, dtype=np.float32)# / 4.0
kernel = np.expand_dims(kernel, axis=0)
kernel = np.expand_dims(kernel, axis=0)
kernel = torch.from_numpy(kernel)
def compute_wpsnr_torch(ref, kernel):
    ''' Function to compute the WPSNR of a frame/block using a torch convolution '''
    ref = np.expand_dims(ref, axis=0)
    ref = np.expand_dims(ref, axis=0)

    ref = torch.from_numpy(ref)
    ref = F.conv2d(ref, kernel, stride=1, padding=0)
    ref = ref.squeeze()
    ref = torch.abs(ref)

    return ref.cpu().detach().numpy()

def get_wpsnr_block_videos(frames):
    ''' Function to compute the WPSNR and XPSNR of a video at the block level '''
    # defince all the parameters from WPSNR metric
    W = 1920
    H = 1080
    BD = 8
    beta = 0.5
    amin = 2 ** (BD - 6)
    amin2 = amin ** 2
    apic = (2 ** BD) * ((3840*2160) / (W*H)) ** 0.5
    xapic = (2 ** (BD+1)) * ((3840*2160) / (W*H)) ** 0.5
    N = 128 * ((3840*2160) / (W*H)) ** 0.5
    N = int(N)
    gamma = 2

    # variables to store the statistics from WPSNR and XPSNR metric
    sum_wk = 0; sum_ak = 0; sum_xk = 0; sum_xak = 0
    all_ak = []; all_xak = []
    cpt = 0
    frame_prec = frames[0]
    for frame in frames: # for each frame in the video
        filtered = compute_wpsnr_torch(frame, kernel) # apply WPSNR filter

        # compute WPSNR statistics
        sum_block = np.sum(filtered) / (filtered.shape[0] * filtered.shape[1])
        ak = max(amin2, sum_block ** 2)
        wk = (apic / ak) ** beta
        sum_wk += wk
        sum_ak += ak
        all_ak.append(ak)

        # compute additional statistics for XPSNR
        diff = gamma * np.abs(frame - frame_prec)
        sum_diff = np.sum(diff) / (diff.shape[0] * diff.shape[1])
        sum_ = sum_diff + sum_block
        xak = max(amin2, sum_ ** 2)
        xk = (xapic / xak) ** beta
        sum_xk += xk
        sum_xak += xak
        all_xak.append(xak)
        cpt += 1
        frame_prec = frame
    return sum_wk / cpt, all_ak, sum_xk / cpt, all_xak

def compute_variance(frame):
    ''' Function to compute the variance of a frame '''
    # for each 8x8 block, compute the variance
    var = 0
    cpt = 0
    step = 8
    for i in range(0, frame.shape[0], step):
        for j in range(0, frame.shape[1], step):
            block = frame[i:i+step, j:j+step]
            var += np.var(block)
            cpt += 1
    return var / cpt

def get_ssim_block_videos(frames):
    ''' Function to compute the SSIM and variance of a video at the block level '''
    all_variance = []
    all_ssim_var = []
    for frame in frames:
        # compute the sum of the variance across all 8x8 blocks
        variance = compute_variance(frame)
        ssim_var = 67.035434 * (1 - math.exp(-0.0021489 * variance)) + 17.492222
        all_variance.append(variance)
        all_ssim_var.append(ssim_var)

    variance = np.mean(all_variance)
    ssim_var = np.mean(all_ssim_var)
    return ssim_var, all_variance


class Opener():
    ''' Class to open a video file and extract the frames '''
    def __init__(self, filename, width=1920, height=1080, yuv_type='yuv420p'):
        self.filename = filename
        self.width = width
        self.height = height
        self.yuv_type = yuv_type
        if self.filename.endswith(".mp4"):
            self.cap = cv2.VideoCapture(self.filename)
            self.yuv_mode = False
        else:
            assert False, "Wrong file format " + self.filename
        return None

    def open_batch_mp4(self, limit=12):
        # open a batch of frames from the video in mp4 format in RGB format
        frames = []
        cap = self.cap
        cpt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            cpt += 1
            if cpt == limit:
                break
        frames = np.array(frames, dtype=np.float32)
        return frames
    
    def open_batch(self, limit=12):
        # open a batch of frames from the video
        return self.open_batch_mp4(limit)

def frames_to_tubes(frames, sp_size=64, addBoarder=False):
    ''' Function to split the frames into 64x64x12 tubes '''
    limit = 12
    tubes, ijk = [], []
    for i in range(0, frames.shape[0], limit):
        for j in range(0, frames.shape[1], sp_size):
            for k in range(0, frames.shape[2], sp_size):
                tube = frames[i:i+limit, j:j+sp_size, k:k+sp_size, :]
                if tube.shape != (limit, sp_size, sp_size, 3):
                    if addBoarder:
                        # if missing row pixel create tube from the last sp_size rows
                        # if missing column pixel create tube from the last sp_size columns
                        # if missing frames create tube from the last frames
                        if tube.shape[0] != limit and tube.shape[1] != sp_size and tube.shape[2] != sp_size:
                            tube = frames[-limit, -sp_size:, -sp_size:, :]
                        elif tube.shape[0] != limit and tube.shape[1] != sp_size:
                            tube = frames[-limit, -sp_size:, k:k+sp_size, :]
                        elif tube.shape[0] != limit and tube.shape[2] != sp_size:
                            tube = frames[-limit, j:j+sp_size, -sp_size:, :]
                        elif tube.shape[1] != sp_size and tube.shape[2] != sp_size:
                            tube = frames[i:i+limit, -sp_size:, -sp_size:, :]
                        elif tube.shape[1] != sp_size:
                            tube = frames[i:i+limit, -sp_size:, k:k+sp_size, :]
                        elif tube.shape[2] != sp_size:
                            tube = frames[i:i+limit, j:j+sp_size, -sp_size:, :]
                        elif tube.shape[0] != limit:
                            tube = frames[-limit:, j:j+sp_size, k:k+sp_size, :]
                        assert tube.shape == (limit, sp_size, sp_size, 3), ("tube shape is not 12xspxspx3", tube.shape)
                    else:
                        continue    
                
                tubes.append(tube)
                ijk.append((i, j, k))
    tubes = np.array(tubes, dtype=np.float32)
    ijk = np.array(ijk)
    return tubes, ijk

def mse_luma(ref_tube, dist_tube):
    ''' Function to compute the MSE of the luma channel between 2 tubes '''
    mse = []
    for i in range(len(ref_tube)):
        ref = ref_tube[i]
        dist = dist_tube[i]
        ref = cv2.cvtColor(np.array(ref, dtype=np.uint8), cv2.COLOR_RGB2YUV)[:,:,0]
        dist = cv2.cvtColor(np.array(dist, dtype=np.uint8), cv2.COLOR_RGB2YUV)[:,:,0]

        mse.append(np.mean((ref.astype(np.float32) - dist.astype(np.float32))**2, axis=(0, 1)))
    return np.mean(mse)

def compute_mse(tubes_ref, tubes_dist, step=5):
    ''' Function to compute the MSE of the luma channel between all the tubes of a sequence '''
    all_mses = []
    for t in tqdm(range(0, len(tubes_ref), step)):
        mse_y = mse_luma(tubes_ref[t], tubes_dist[t])
        all_mses.append(mse_y)
    all_mses = np.array(all_mses)
    return all_mses

def get_other_features(tubes_ref):
    ''' Function to compute the SSIM, WPSNR and XWPSNR of a video at the block level '''
    ssim_vars = []; wks = []; xwks = []
    for tube in tubes_ref: # for each tubes in the sequence
        frames = []
        for f in range(len(tube)):
            frames.append(cv2.cvtColor(np.array(tube[f], dtype=np.uint8), cv2.COLOR_RGB2YUV)[:,:,0])
        frames = np.array(frames, dtype=np.float32)

        ssim_var, variance = get_ssim_block_videos(frames)
        wk, ak, xwk, xak = get_wpsnr_block_videos(frames)
        ssim_vars.append(ssim_var); wks.append(wk); xwks.append(xwk)
    ssim_vars = np.array(ssim_vars); wks = np.array(wks); xwks = np.array(xwks)
    # log sum of ssim_var
    log_sum_ssim_var = np.log(ssim_vars).sum() / len(ssim_vars)
    geo_mean_ssim_var = np.exp(log_sum_ssim_var)
    norm_ssim_vars = ssim_vars / geo_mean_ssim_var #same as in libaom
    return 1/ssim_vars, norm_ssim_vars, wks, xwks

def saliency_to_string(pred_saliency):
    ''' Function to convert saliency array to a string to store in a txt file 
        This txt file will be read by the libaom encoder to use the weight in RDO process '''
    lines = ''
    for i in range(len(pred_saliency)):
        lines += ','.join([str(c) for c in pred_saliency[i][:]]) + '\n'
    return lines

def store_saliency_in_txt(pred_saliency, filename, deltaq=None):
    ''' Function to convert saliency array to a string to store in a txt file 
        This txt file will be read by the libaom encoder to use the weight in RDO process '''
    lines = ''
    if deltaq is not None:
        lines += ','.join([str(c) for c in deltaq]) + '\n'
    lines += saliency_to_string(pred_saliency)
    with open(filename, 'w') as f:
        f.write(lines)
    return None

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict VCIP models')
    argparser.add_argument('--dist_path', type=str, help='Path to distorted video')
    argparser.add_argument('--ref_path', type=str, help='Path to reference video')
    argparser.add_argument('--width', type=int, default=0, help='Width of the video')
    argparser.add_argument('--height', type=int, default=0, help='Height of the video')
    argparser.add_argument('--score_file', type=str, default="", help='version tag')

    args = argparser.parse_args()
    model_path = ""

    nb_layers = 0
    INCLUDE_WK_SSIM = 5
    INCLUDE_STD = 0
    nb_ref_features = 0


    fileout = args.score_file

    # Define the parameters of the neural network
    L = 2
    Hl = 64 #128
    input_size = 2
    output_size = 4
    hidden_layers = L
    hidden_neurons = Hl
    activation_func = 'leakyrelu'  # or 'leakyrelu' or 'tanh'
    batch_norm = True  # [True, False]
    dropout = 0.01
    learning_rate = 0.0001
    num_epochs = 20000
    l2_reg_weight = 0.001 # 0.001

    # Create the neural network and load the weights
    model = NeuralNetwork(input_size, output_size, hidden_layers, hidden_neurons, activation_func, batch_norm, dropout)
    deep_model_name = f"deep_model.pth"
    model.load_state_dict(torch.load(deep_model_name))
    model = model.eval()

    # parse the parameters of the command
    t1 = time()
    filename_ref = args.ref_path
    filename_dist = args.dist_path
    width = args.width
    height = args.height
    if filename_ref.endswith(".yuv") or filename_dist.endswith(".yuv"):
        if width == 0 or height == 0:
            print("Please specify width and height of the video")
            exit()

    # define workspace and tools
    workspace = "../external_data_for_encode"
    encodes_folder = "../encodes"
    #FFPROBE = "/home/pastor/FFmpeg/ffprobe"
    FFPROBE = "ffprobe"

    # count the number of frames
    cmd = f"{FFPROBE} -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {filename_dist}"

    # get the number of frames in the video by running ffprobe
    nb_frames = int(subprocess.check_output(cmd, shell=True).decode("utf-8").strip())
    print("nb_frames", nb_frames)


    # define the video openers    
    op_ref = Opener(filename_ref, width=width, height=height)
    op_dist = Opener(filename_dist, width=width, height=height)

    scores = []; pds = []; mses_ = []
    scores_raw = []; pds_raw = []
    scores_var = []; pds_var = []
    scores_wk = []; pds_wk = []
    scores_xwk = []; pds_xwk = []
    debug_plot = False
    mses = []; mses_weighted = []; ssim_vars = []; wks = []; xwks = []
    wmse_slopes = []; ssim_vars_slopes = []; wks_slopes = []; xwks_slopes = []
    for i in range(nb_frames // 12):
        print((i, nb_frames // 12))
        # read 12 frames
        frames_ref = op_ref.open_batch(12)
        frames_dist = op_dist.open_batch(12)
        yy = frames_ref.shape[1] // 64; xx = frames_ref.shape[2] // 64
        if yy != frames_ref.shape[1] / 64:
            yy += 1
        if xx != frames_ref.shape[2] / 64:
            xx += 1
        W, H = frames_ref.shape[2] // 16, frames_ref.shape[1] // 16
        if W != frames_ref.shape[2] / 16:
            W += 1
        if H != frames_ref.shape[1] / 16:
            H += 1

        # split frames into 64x64x12 tubes
        tubes_ref, ijk = frames_to_tubes(frames_ref, addBoarder=True)
        tubes_dist, ijk = frames_to_tubes(frames_dist, addBoarder=True)

        step = 1
        print("computing mse", tubes_ref.shape, tubes_dist.shape)
        all_mses = compute_mse(tubes_ref, tubes_dist, step=step)

        # compute the other features
        all_ssim_vars, all_ssim_vars_frame_norm, all_wks, all_xwks = get_other_features(tubes_ref[::step])
        weighted_ssim = all_mses * 1 / all_ssim_vars
        weighted_ssim_frame_norm = all_mses * 1 / all_ssim_vars_frame_norm 
        weighted_wk = all_mses * all_wks
        weighted_xwk = all_mses * all_xwks

        all_acti_ref, all_acti_ref_std = None, None 
        X = all_acti_ref; X_std = all_acti_ref_std

        if INCLUDE_WK_SSIM:
            if INCLUDE_WK_SSIM == 1:
                features = np.concatenate((np.expand_dims(all_wks, axis=1), np.expand_dims(all_ssim_vars, axis=1)), axis=1) # concatenate the features
            elif INCLUDE_WK_SSIM == 2:
                features = np.expand_dims(all_wks, axis=1)
            elif INCLUDE_WK_SSIM == 3:
                features = np.expand_dims(all_ssim_vars, axis=1)
            elif INCLUDE_WK_SSIM == 4:
                features = np.expand_dims(all_xwks, axis=1)
            elif INCLUDE_WK_SSIM == 5:
                features = np.concatenate((np.expand_dims(all_xwks, axis=1), np.expand_dims(all_ssim_vars, axis=1)), axis=1) # concatenate the features
            else:
                assert False, ("INCLUDE_WK_SSIM not recognized", INCLUDE_WK_SSIM)
            all_mses = np.array(all_mses)

            if nb_layers is None or nb_layers > 0:
                X = np.concatenate((features, X_std[:,:INCLUDE_STD], X), axis=1)[:,:nb_ref_features - 1]
            else:
                X = features
        else:
            X = np.concatenate((X_std[:,:INCLUDE_STD], X), axis=1)[:,:nb_ref_features - 1]

        # pass the data in the neural network
        outputs = model(torch.tensor(X).float())
        #slope_exp_a = outputs[:, 0].detach().numpy()
        #slope_exp_b = outputs[:, 1].detach().numpy()
        all_mses = outputs[:, 2].detach().numpy()
        svr_pred = outputs[:, 3].detach().numpy()
        svr_pred_map = svr_pred.reshape((yy, xx))
        all_mses_ = all_mses.reshape((yy, xx))

        svr_pred_map[svr_pred_map < 1e-3] = 1e-3 # region where the slope is almost 0 is set to 1e-3 to avoid too much lambda correction (over compression of the image)
        inv_svr_pred_map = 1/svr_pred_map
        mean_geo_inv_svr_pred_map = np.exp(np.log(inv_svr_pred_map).mean())
        inv_svr_pred_map_ = (inv_svr_pred_map / mean_geo_inv_svr_pred_map) ** 1

        for _ in range(5): # 5 iterations of the correction to get a better estimation of the geo mean
            mean_geo_inv_svr_pred_map = np.exp(np.log(inv_svr_pred_map_).mean())
            inv_svr_pred_map_ = (inv_svr_pred_map_ / mean_geo_inv_svr_pred_map)

        inv_svr_pred_map_ = np.clip(inv_svr_pred_map_, 0.6, 5*0.6)                

        svr_pred_map_ = 1 / inv_svr_pred_map_
        scores_alpha = cv2.resize(svr_pred_map_, (W, H), interpolation=cv2.INTER_NEAREST)

        id = "21"
        sequence_name = filename_dist.split('/')[-1].split('_')[0]
        path_to_folder = f"{workspace}/{sequence_name}/dl2_rawlin{id}/"
        # create the folder to store the map
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        for j in range(13): # put back to 12 frames (since I used that to have extract txt required during the encoding)
            fid = i * 12 + j
            fid = str(fid+1).zfill(5)
            filename = path_to_folder + f'frame{fid}_alpha.txt'
            print(filename)
            store_saliency_in_txt(1 / scores_alpha, filename)
        
        print(svr_pred.shape, np.mean(svr_pred))

    print("Done")