# ////////////////////////////////////////////////////////////////////////////
# //  This file is part of NIID-Net. For more information
# //  see <https://github.com/zju3dv/NIID-Net>.
# //  If you use this code, please cite the corresponding publications as
# //  listed on the above website.
# //
# //  Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.
# //
# //  Permission to use, copy, modify and distribute this software and its
# //  documentation for educational, research and non-profit purposes only.
# //
# //  The above copyright notice and this permission notice shall be included in all
# //  copies or substantial portions of the Software.
# //
# //  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# // SOFTWARE.
# ////////////////////////////////////////////////////////////////////////////


import os
import os.path
import random
import time
import argparse
import multiprocessing


import open3d as o3d  # import open3d(0.7.0) at the end causes segmentation fault, don't know why
from matplotlib import pyplot as plt
from skimage.transform import resize
from PIL import Image
import pandas as pd
import numpy as np


cmap = plt.cm.jet
bit_16_img = 65535.0
bit_8_img = 255.0


class NYU_RGB_Camera_Params(object):
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02
    H = 480
    W = 640


class ScanNet_RGB_Camera_Params(object):
    fx = 517.97
    fy = 517.97
    cx = 320
    cy = 240
    H = 480
    W = 640


def get_NYU_valid_mask():
    mask = np.zeros((480, 640), dtype=np.float32)
    mask[44:470, 40:600] = 1.0
    return mask


def compute_3D_points_map(depth, camera_params=NYU_RGB_Camera_Params, eps=1e-6):
    H_d = depth.shape[0]
    W_d = depth.shape[1]
    assert camera_params.H // H_d == camera_params.W // W_d, 'H/W ratio of depth map is not consistent with provided camera matrix'

    # resize depth map
    valid_mask = (depth > eps).astype(np.float32)
    depth = resize(depth, (camera_params.H, camera_params.W), order=1, preserve_range=True, mode='constant', anti_aliasing=True).squeeze()
    valid_mask = resize(valid_mask, (camera_params.H, camera_params.W), order=1, preserve_range=True, mode='constant', anti_aliasing=False).squeeze()
    valid_mask = (valid_mask > 0.999).astype(np.float32)

    # project to camera space
    x = np.arange(1, depth.shape[1] + 1, 1)
    y = np.arange(1, depth.shape[0] + 1, 1)
    xx, yy = np.meshgrid(x, y)
    z3 = depth
    x3 = (xx - camera_params.cx) * depth / camera_params.fx
    y3 = (yy - camera_params.cy) * depth / camera_params.fy
    points3d = np.dstack((x3[:], y3[:], z3[:])).astype(np.float32)
    valid_mask = valid_mask[..., np.newaxis]
    points3d = np.multiply(points3d, valid_mask)
    return points3d, valid_mask


def estimate_surface_normal(depth, rgb_img=None, _mask=None, camera="NYUv2", search_params=(1.0, 200), visualize=False):
    img_shape = depth.shape[0:2]

    # compute 3d point cloud
    ## select camera intrinsic parameters
    if camera == "NYUv2":
        camera_params = NYU_RGB_Camera_Params
    elif camera == "ScanNet":
        camera_params = ScanNet_RGB_Camera_Params
    else:
        print("Error: not supports camera %s" % camera)
        return
    ## 3D point cloud
    points_3d, valid_mask = compute_3D_points_map(depth, camera_params)
    ## merge valid mask
    valid_mask = np.min(valid_mask, axis=2)
    if _mask is not None:
        valid_mask *= _mask
    select_mask = valid_mask > 0.5

    # convert to o3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d[select_mask])
    if rgb_img is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb_img[select_mask])

    # estimate normal
    radius, max_nn = search_params
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                          max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    # Code for open3d 0.7.0
    # o3d.geometry.estimate_normals(
    #     pcd,
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
    #                                                       max_nn=max_nn))
    # o3d.geometry.orient_normals_towards_camera_location(pcd, np.array([0, 0, 0]))
    normal_map = np.zeros((img_shape[0], img_shape[1], 3))
    normal_map[select_mask] = np.asarray(pcd.normals)

    # visualize
    if visualize:
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd])

    return normal_map, valid_mask


class SurfaceNormalEstimator(object):
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        csv_file = os.path.join(dataset_dir, "nyu2_train.csv")
        self.frame = pd.read_csv(csv_file, header=None)
        self.frame_len = len(self.frame)
        self.search_params = (0.1, 3000)
        self.camera = "NYUv2"
        print('size of NYUv2 train: %d' % self.frame_len)

    def estimate(self, index):
        # data_time = time.time()
        # Data path
        image_name = os.path.join(self.dataset_dir, self.frame.iloc[index, 0][5:])
        depth_name = os.path.join(self.dataset_dir, self.frame.iloc[index, 1][5:])

        # split file name and path
        out_path, file_name = os.path.split(depth_name)
        file_name = file_name[:-4]
        out_path = out_path[out_path.find("nyu2_train/")+len("nyu2_train/"): ]
        out_path = os.path.join(self.output_dir, out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)


        # Read data
        rgb_img = Image.open(image_name)
        depth_img = Image.open(depth_name)
        rgb_img = np.array(rgb_img) / bit_8_img
        depth_img = np.array(depth_img) / bit_8_img * 10.0
        mask = get_NYU_valid_mask()

        # Estimate surface normal
        normal_map, mask = estimate_surface_normal(depth_img, rgb_img, mask, self.camera, self.search_params, False)
        normal_vis = (-normal_map.clip(min=-1, max=1) + 1) / 2.0 * bit_8_img
        Image.fromarray(normal_vis.astype(np.uint8), mode="RGB").save(os.path.join(out_path, file_name + "_normal.png"))
        # np.save(os.path.join(out_path, file_name + "_normal.npy"), normal_map)
        # mask_img = Image.fromarray((mask*255).astype('uint8'))
        # mask_img.save(os.path.join(out_path, file_name + "_mask.jpg"))

        # visualization
        vis_normal = (-normal_map.clip(min=-1, max=1) + 1) / 2.0
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1).set_title("rgb_img")
        plt.imshow(rgb_img)
        plt.subplot(1, 3, 2).set_title("depth")
        plt.imshow(depth_img, cmap=cmap)
        plt.subplot(1, 3, 3).set_title("normal")
        plt.imshow(vis_normal)
        # plt.show()
        plt.savefig(os.path.join(out_path, file_name + "_vis.jpg"), bbox_inches='tight')
        plt.close()

        # print('process %d/%d %s: %lf' % (index+1, len(self.frame), depth_name, time.time()-data_time))
        if index % 50 == 0:
            print('process %d/%d %s' % (index+1, len(self.frame), depth_name))
        return index

    def run(self, num_workers):
        print('the number of used process %d' % num_workers)
        pool = multiprocessing.Pool(processes=num_workers)

        cnt = 0
        data_time = time.time()
        for index in pool.imap_unordered(self.estimate, range(self.frame_len)):
            cnt += 1
            if cnt >= 50:
                print("time per image: %.5lf " % ((time.time() - data_time) / cnt))
                data_time = time.time()
                cnt = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default='~/Datasets/depth/NYU_v2',
        metavar="FILE",
        help="Path to NYU_v2 dataset",
        type=str,
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        metavar="N",
        help="number of multi-processes",
        type=int,
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = os.path.join(dataset_dir, "nyu2_train_normal")
    num_workers = args.num_workers
    print('dataset_dir: ', dataset_dir)
    print('output_dir: ', output_dir)
    print('num_workers: ', num_workers)
    normal_estimator = SurfaceNormalEstimator(dataset_dir, output_dir)
    normal_estimator.run(num_workers)
