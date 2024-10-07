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
import pickle
from skimage import io
from PIL import Image


def extract_IIW_images(dir_path, IIW_dir='./dataset/CGIntrinsics/IIW/'):
    """ Extract test images of the IIW dataset into the directory

    :param dir_path: output directory
    :param IIW_dir: the directory of the IIW dataset
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    IIW_test_list = IIW_dir + 'test_list/'
    file_name = IIW_test_list + "img_batch.p"
    images_list = pickle.load(open(file_name, "rb"), encoding='iso-8859-1')
    for mode in range(0, 3):
        num_imgs = len(images_list[mode])
        for index in range(num_imgs):
            img_name = (images_list[mode][index].split('/'))[-1][:-3]
            img_path = IIW_dir + 'data/' + img_name
            img = io.imread(img_path)
            io.imsave(os.path.join(dir_path, img_name), img)
            print('save %d/%d: %s' % (index+1, num_imgs, os.path.join(dir_path, img_name)))


def extract_SAW_images(dir_path, SAW_dir = './dataset/CGIntrinsics/SAW/'):
    """ Extract test images of the SAW dataset into the directory

    :param dir_path: output directory
    :param SAW_dir: the directory of the SAW dataset
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    img_dir = SAW_dir + 'saw_images_512/'
    splits_dir = SAW_dir + 'saw_splits/'
    dataset_split = 'E'

    from test import saw_utils
    photo_ids = saw_utils.load_photo_ids_for_split(splits_dir, dataset_split)
    total_num_img = len(photo_ids)
    for idx, photo_id in enumerate(photo_ids):
        img_name = str(photo_id) + ".png"
        img_path = img_dir + img_name
        img = Image.open(img_path)
        img.save(os.path.join(dir_path, img_name))
        print('save %d/%d: %s' % (idx + 1, total_num_img, os.path.join(dir_path, img_name)))


if __name__ == '__main__':
    extract_IIW_images('./decompose/test_IIW/')
    # extract_SAW_images('./decompose/test_SAW/')
