# Real-time, Low-cost, and Automated Estimation of Cementitious Sorptivity via Computer Vision


###  Hossein Kabir, Jordan Wu, Sunav Dahal, Tony Joo, Nishant Garg

University of Illinois at Urbana-Champaign, Urbana, IL, USA

##
### System requirements

-    Operating System: Compatible with systems that support [Python](https://www.python.org) and [Jupyter Notebooks](https://jupyter.org).
  
-    Python Version: Compatible with [Python 3.x](https://www.python.org/download/releases/3.0/).
  
-    Hardware Requirements: Requires a [CUDA-capable](https://en.wikipedia.org/wiki/CUDA) GPU to ensure model training can leverage 
    GPU acceleration as indicated by the assertion [torch.cuda.is_available](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html).

##
### Step-by-Step Instructions for Running Colab Notebook

- Tested Version: The notebook has been configured to work on Google Colab with specific paths
    set for Google Drive. Thus, it's tested on the Google Colab environment with
    internet connectivity and access to Google Drive.
  
- Access to Google Drive with at least 1 GB of free space.
  
- Structure your project (under My Drive folder) as follows:
  

  | Folder Name | Description |
  | ------ | ------ |
  | [**Colab Notebook**](https://drive.google.com/drive/folders/1VXraqL6XG5al7IzVSfAvfs7rNyPc81K6?usp=sharing)|stores the main Jupyter Notebook|
  | [**checkpoints**](https://drive.google.com/drive/folders/14JEJopo-M52N12BDNKt9hH_71cbjDMsC?usp=sharing)|stores the weigh matrices from the latest epoch to do prob plot predictions|
  | [**src**](https://drive.google.com/drive/folders/1h4KCDqu05fEYjzmGErJZnSE9ieeHnHZ6?usp=sharing) | accomodates custom scripts (models.py, dataset_loader.py, evaluation.py, util.py, visualize.py) |
  | [**visualization**](https://drive.google.com/drive/folders/1Im6fSw2cN3AlGQRpsc5DogKmQNVvif29?usp=sharing) | stores the predicted binary masks |
  | [**dataset.zip**](https://drive.google.com/file/d/1D5C6k-oRo9EgWlSMo-OrTfVPaF8FGehr/view?usp=sharing) | includes the train and test (arbitrary) set images|
  | [**predictions.zip**](https://drive.google.com/file/d/1R6w2CCwDX6SYWtl6j8tSevRXT1v5yXFH/view?usp=sharing) | includes the compressed predicted prob plots|

- Your Google Drive directory should be organized as shown below:

  <img src="https://github.com/hosseinkabiruiuc/Sorptivity-via-Computer-Vision/blob/main/src/Google%20Drive_%20Dircect.png" alt="Google Drive Directory" width="60%">

- Open the [model_compile.ipynb](https://drive.google.com/file/d/1OBWePqsPNm9ZQ0nfNJV0asDuayekJtoU/view?usp=sharing) Jupyter notebook in Google Colab.
  
- Execute the cells sequentially, which will involve mounting the drive, loading data, training models, and visualizing results.

- Execution Time:
  The time to run the notebook will heavily depend on the dataset size and your hardware, especially whether a GPU is available or not. Using [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/), it should take ~ 2 hours (with full dataset, i.e., few thousands images) to train the FPN model and less than 10 seconds to do [water predictions](https://drive.google.com/drive/folders/1Im6fSw2cN3AlGQRpsc5DogKmQNVvif29?usp=sharing). 

- Prior to visualizing the segmented mask, the predicted [probability plots](https://drive.google.com/drive/folders/1RBsCfsSSezS4DA9j9n7E43wF65yUBHjg?usp=sharing) folder which is yield after training the [FPN or Mask R-CNN models](https://drive.google.com/drive/folders/1YjN6jhbAd2zVVBGiKyQb4YMCMZFE1qKw?usp=sharing), should be [compressed](https://drive.google.com/file/d/1R6w2CCwDX6SYWtl6j8tSevRXT1v5yXFH/view?usp=sharing) and placed in the main directory.

- Model output: The real-time water prediction is shown as follows: 

  <img src="https://github.com/hosseinkabiruiuc/Sorptivity-via-Computer-Vision/blob/main/visualization/output.gif" width="50%" alt="GIF Description">

##
### Software Dependencies

   | Libraries | Justification |
   | ------ | ------ |
   | [**PIL**](https://pillow.readthedocs.io/en/stable/)|used for opening, manipulating, and saving image file formats|
   | [**torch**](https://pypi.org/project/torch/)|implemented for neural network models and GPU computing|
   | [**torchvision**](https://pytorch.org/vision/stable/index.html)| imported for image transformations and dataset loaders|
   | [**pytorch lightning**](https://lightning.ai/docs/pytorch/stable/)|applied for more structured and cleaner training loops|
   | [**segmentation models pytorch**](https://pypi.org/project/segmentation-models-pytorch/0.0.3/)|employed for pretrained segmentation models|
   | [**matplotlib**](https://matplotlib.org)|deployed for plotting graphs and visualizations|
   | [**numpy**](https://numpy.org)|adopted for numerical operations|
   | [**plotly**](https://plotly.com/python/)|leveraged for interactive visualizations|
   | [**pandas**](https://pandas.pydata.org)|activated for data manipulation and reading/writing CSV files|
   | [**re**](https://docs.python.org/3/library/re.html)|incorporated for regular expression operations|
   | [**pathlib**](https://docs.python.org/3/library/pathlib.html)|used for system path operations|
   | [**io**](https://docs.python.org/3/library/io.html)|implemented for handling IO operations|

##
### Custom Scripts

   | Libraries | Explanation |
   | ------ | ------ |
   | [**models.py**](https://drive.google.com/file/d/1T6sRbHbAepBUVcEb64hMdHqAuyk46pvx/view?usp=sharing)|defines a PyTorch Lightning model for image segmentation, incorporating training, validation, and model management features like checkpointing and early stopping|
   | [**dataset_loader.py**](https://drive.google.com/file/d/1fbbW1KSw90h59OQxc6bTUrAApJmLZ78Z/view?usp=sharing)|manages the loading and preprocessing of image datasets for segmentation tasks, supporting both real and synthetic data|
   | [**evaluation.py**](https://drive.google.com/file/d/1-fuBl38ZN8SW4F6B7SXVs0ycbRhM4YVn/view?usp=sharing)|calculates the Intersection over Union (IoU) metric for evaluating model performance in tasks involving segmentation or object detection|
   | [**util.py**](https://drive.google.com/file/d/1cZ2r58EHfoNZvYtvih7LMPVWQeU1yY25/view?usp=sharing)|provides utility functions for file and directory management, and data manipulation in Python|
   | [**visualize.py**](https://drive.google.com/file/d/1bmumXqfe-h_ZtAUqhzmdIoWugE_AmfYU/view?usp=sharing)|provides functions for visualizing segmentation masks and IoU histograms to assess model predictions against ground truths|

