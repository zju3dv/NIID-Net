# NIID-Net: Adapting Surface Normal Knowledge for Intrinsic Image Decomposition in Indoor Scenes
[[paper]](https://ieeexplore.ieee.org/document/9199573) 
[[supplement]](http://www.cad.zju.edu.cn/home/gfzhang/papers/NIID-Net/NIID-Net-supple.pdf)
[[presentation]](https://youtu.be/BvoYwCdzoZU)
[[demo]](https://youtu.be/0MadIlfqles) 

![architecture](./assets/NIID-Net.png)


Updates
-
+ 06/October/2024: Modified the visualization of output images.
+ 16/April/2023: Migrated to Pytorch 1.7.1.

Dependencies
-
+ Python 3.6
+ PyTorch 1.7.1 (original [version](https://github.com/zju3dv/NIID-Net/tree/pytorch_0.3.1) using PyTorch 0.3.1)
+ torchvision 0.8.2
+ [Visdom](https://github.com/facebookresearch/visdom) 0.1.8.9 
+ We provide the ```tools/install.txt``` file for other dependencies.

Datasets
-
#### Intrinsic image datasets
+ Follow [CGIntrinsics](https://github.com/zhengqili/CGIntrinsics) to download CGI, IIW and SAW datasets. 
Note that Z. Li and N. Snavely augment the original [IIW](http://opensurfaces.cs.cornell.edu/intrinsic/#) and [SAW](http://opensurfaces.cs.cornell.edu/saw/) datasets.
+ You do not need to download the CGI dataset if you are not going to train the model.
+ Put the datasets in the ```./dataset/``` folder. The final directory structure:
    ```
    NIID-Net project
    |---README.md
    |---...
    |---dataset
        |---CGIntrinsics
            |---intrinsics_final
            |   |---images   
            |   |---rendered
            |   |---...
            |---IIW
            |   |---data
            |   |---test_list
            |   |---...
            |---SAW
                |---saw_images_512
                |---saw_pixel_labels
                |---saw_splits
                |---train_list
    ```

#### Depth/surface normal dataset
+ Follow [Revisiting_Single_Depth_Estimation](https://github.com/JunjH/Revisiting_Single_Depth_Estimation) to 
download their extracted NYU-v2 subset.
+ Unzip ```data.zip``` and rename the directory as ```NYU_v2```
+ To compute surface normal maps,
  + install ```open3d==0.8.0``` 
  + ```python ./tools/data_preprocess_normal.py --dataset_dir {NYU_v2 dataset path}  --num_workers {number of processes}```
  + For some historical reasons, the surface normal maps (*_normal.png) generated are in a left-handed coordinate system: 
    x: left, y: up, z: from screen to viewer.

Running
-
+ ##### Configuration
  + ```options/config.py``` is the configuration file:
    + ```TestOptions``` for test
    + ```TrainIIDOptions``` for training the IID-Net
  + Some variables may need to be modified:
    ```
      dataset_root #
      checkpoints_dir # visualized results will be saved here
      offline # if you do not need Visdom, set it True
      pretrained_file #
      gpu_devices # the indexes of GPU devices, or set None to run CPU version 
      batch_size_intrinsics # batch size for training on the CGIntrinsics dataset
    ```
  + Note that only test mode supports CPU version (with ```gpu_devices=None```). 
  We recommend you to use the **GPU** version.
+ ##### Pre-trained model
  + [Google Drive](https://drive.google.com/file/d/160NzDEmC8okb6vgTNTyzmhaYa-Lqo-Ft/view?usp=sharing)
    (or [Baidu Net Disk](https://pan.baidu.com/s/1n45ZwuYZpUA8vp-9V-ca9Q) with code ```uj3n```)
  + Download the pre-trained model to ```./pretrained_model/```
    
+ ##### Demo
  + ```python decompose.py```
    + The visualized intrinsic images are in linear RGB space. 
      For sRGB images, add option ```--save_srgb```.
    + The input images are resized to a default size, 
      you can change the size by setting ```--resize_to_specified_size 1024```.
    + This script automatically converts the output surface normals to this popular camera system:
      + x: right, y: up, z: from screen to viewer


+ ##### Test
  + ```python evaluate.py --func <function>```
    + ```<function>``` can be ```test_iiw```, ```test_saw```
    + The default output directory is ```./checkpoints/```
  
+ ##### Model size calculation
  + ```python evaluate.py --func model_size```
  
+ ##### Train
  + ```python train_IID.py```


Results
- 
#### Precomputed results
- We provide precomputed results on the SAW and IIW test sets:
  - SAW:
    - Visualized images (linear RGB space): [SAW_pred_imgs.zip](https://drive.google.com/file/d/10DOCkHGG4cWFyVwQ93eRAZhzYr1VT4Cf/view?usp=drive_link).
    - Raw predictions (linear RGB space, float32): [SAW_raw_pred.zip](https://drive.google.com/file/d/10OY2OtAqDd9l4COEZIF-lWsiu9L939Vy/view?usp=sharing).
    - Precision-recall measurements 
    ([precision-recall_curves.zip](https://drive.google.com/file/d/1WhxxN5sSVLLoet1ruk9VHzh_r-WhfRfs/view?usp=sharing)), 
    which can be used to draw the precision-recall curves. 
  - IIW:
    - Visualized and raw predictions (linear RGB space): [IIW_test.zip](https://drive.google.com/file/d/10NHd0QwuY6JxDN3OGy7gzwyMpRtkRxgs/view?usp=sharing).
- For comparison on some applications (e.g., image editing), we recommend using the original float32 output of the network
instead of the 8-bit low-precision visualized images.


#### Comparison
![comparison](./assets/comparison.jpg)

#### Image sequence editing
![editing](./assets/demo1.jpg)

Acknowledgements
-
We have used/modified codes from the following projects:
  + [CGIntrinsics](https://github.com/zhengqili/CGIntrinsics):
    + codes for loading data from the CGI, SAW and IIW datasets in ```./data/intrinsics/```.
    + codes for evaluating shading and reflectance estimation in ```./test/```.
    (These codes are originally provided by [IIW](http://opensurfaces.cs.cornell.edu/intrinsic/#) 
    and [SAW](http://opensurfaces.cs.cornell.edu/saw/)) 
  + [Revisiting_Single_Depth_Estimation](https://github.com/JunjH/Revisiting_Single_Depth_Estimation):
    + the network structure of normal estimation module in ```./models/Hu_nets/```
    

Citation
-
If you find this code useful for your research, please cite:
  ```
  @article{luo2020niid,
    title={NIID-Net: Adapting Surface Normal Knowledge for Intrinsic Image Decomposition in Indoor Scenes},
    author={Luo, Jundan and Huang, Zhaoyang and Li, Yijin and Zhou, Xiaowei and Zhang, Guofeng and Bao, Hujun},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    year={2020},
    publisher={IEEE}
  }
  ```

Copyright
-
```
  Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.

  Permission to use, copy, modify and distribute this software and its
  documentation for educational, research and non-profit purposes only.

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
```

Contact
-
Please open an issue or contact Jundan Luo (<jundanluo22@gmail.com>) if you have any questions or any feedback.