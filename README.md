# Real-time, Low-cost, and Automated Estimation of Cementitious Sorptivity via Computer Vision
###  Hossein Kabir, Jordan Wu, Sunav Dahal, Tony Joo, Nishant Garg

## Google Drive Setup for Colab Notebook

### Step-by-Step Instructions

- Access to Google Drive with at least 1 GB of free space.
- Structure your project (under My Drive folder) as follows:

| Folder Name | Description |
| ------ | ------ |
| [Colab Notebook](https://drive.google.com/drive/folders/1VXraqL6XG5al7IzVSfAvfs7rNyPc81K6?usp=sharing)|stores the main Jupyter Notebook|
| [checkpoints](https://drive.google.com/drive/folders/14JEJopo-M52N12BDNKt9hH_71cbjDMsC?usp=sharing)|stores the weigh matrices from the latest epoch to do prob plot predictions|
| [src](https://drive.google.com/drive/folders/1h4KCDqu05fEYjzmGErJZnSE9ieeHnHZ6?usp=sharing) | accomodates custom scripts (models.py, dataset_loader.py, evaluation.py, util.py, visualize.py) |
| [visualization](https://drive.google.com/drive/folders/1Im6fSw2cN3AlGQRpsc5DogKmQNVvif29?usp=sharing) | stores the predicted binary masks |
| [dataset.zip](https://drive.google.com/file/d/1D5C6k-oRo9EgWlSMo-OrTfVPaF8FGehr/view?usp=sharing) | includes the train and test (arbitrary) set images|
| [predictions.zip](https://drive.google.com/file/d/1R6w2CCwDX6SYWtl6j8tSevRXT1v5yXFH/view?usp=sharing) | includes the compressed predicted prob plots|

- Your Google Drive directory should be organized as shown below::

<img src="https://github.com/hosseinkabiruiuc/Sorptivity-via-Computer-Vision/blob/main/src/Google%20Drive_%20Dircect.png" alt="Google Drive Directory" width="70%">

- Open the [model_compile.ipynb](https://drive.google.com/file/d/1OBWePqsPNm9ZQ0nfNJV0asDuayekJtoU/view?usp=sharing) Jupyter notebook in Google Colab.
  
- Execute the cells sequentially, which will involve mounting the drive, loading data, training models, and visualizing results.

- Execution Time:
  The time to run the notebook will heavily depend on the dataset size and your hardware, especially whether a GPU is available or not. Using [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/), It should take ~ 2 hours (with full dataset, i.e., few thousands images) to train the FPN model and less than 1 minute to do [water predictions](https://drive.google.com/drive/folders/1Im6fSw2cN3AlGQRpsc5DogKmQNVvif29?usp=sharing). 

- Prior to visualizing the segmented mask, the predicted [probability plots](https://drive.google.com/drive/folders/1RBsCfsSSezS4DA9j9n7E43wF65yUBHjg?usp=sharing) folder which is yield after the [FPN or Mask R-CNN model traning](https://drive.google.com/drive/folders/1YjN6jhbAd2zVVBGiKyQb4YMCMZFE1qKw?usp=sharing), should be [compressed](https://drive.google.com/file/d/1R6w2CCwDX6SYWtl6j8tSevRXT1v5yXFH/view?usp=sharing) and placed in the main directory.  


### ✨System requirements✨

-    Operating System: Compatible with systems that support [Python](https://www.python.org) and [Jupyter Notebooks](https://jupyter.org).
-    Python Version: Compatible with [Python 3.x](https://www.python.org/download/releases/3.0/).
-    Hardware Requirements: Requires a [CUDA-capable](https://en.wikipedia.org/wiki/CUDA) GPU to ensure model training can leverage 
    GPU acceleration as indicated by the assertion [torch.cuda.is_available](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html).

### ✨Software Dependencies✨

| Libraries | Description |
| ------ | ------ |
| [**PIL**](https://pillow.readthedocs.io/en/stable/)|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|
| [****]()|adds support for opening, manipulating, and saving image file formats|



-    **torch**: Main library for neural network models and GPU computing.
-    **torchvision**: For image transformations and dataset loaders.
-    **pytorch lightning**: For more structured and cleaner training loops.
-    **matplotlib**: For plotting graphs and visualizations.
-    **numpy**: For numerical operations.
-    **segmentation models pytorch**: For pretrained segmentation models.
-    **plotly**: For interactive visualizations.
-    **pandas**: For data manipulation and reading/writing CSV files.
-    **re**: For regular expression operations.
-    **pathlib**: For system path operations.
-    **io**: For handling IO operations.
-    **other custom scripts** (models.py, dataset_loader.py, evaluation.py, util.py, visualize.py) are used, 
    so ensure these are included in the system path or installation directory.

### ✨Tested Versions✨

-    The notebook has been configured to work on Google Colab with specific paths
    set for Google Drive. Thus, it's tested on the Google Colab environment with
    internet connectivity and access to Google Drive.
        

### ✨Typical Execution Time✨

-    Local Machine:
    Installation of Python and dependencies should take about 15-30 minutes on a normal
    desktop computer, depending on your internet connection.
    


## Output

<img src="https://github.com/hosseinkabiruiuc/Sorptivity-via-Computer-Vision/blob/main/Outputs/output.gif?raw=true" width="50%" alt="GIF Description">


[Link to dataset](https://drive.google.com/file/d/1uiP14oo8_4OTx6sBgO-uor0SxDhtsdxG/view?usp=sharing)
