# Real-time, Low-cost, and Automated Estimation of Cementitious Sorptivity via Computer Vision
###  Hossein Kabir, Jordan Wu, Sunav Dahal, Tony Joo, Nishant Garg


<img src="https://github.com/hosseinkabiruiuc/Sorptivity-via-Computer-Vision/blob/main/Outputs/output.gif?raw=true" width="50%" alt="GIF Description">



### ✨System requirements✨

    Operating System: Compatible with systems that support Python and Jupyter notebooks.
    Python Version: Compatible with Python 3.x.
    Hardware Requirements: Requires a CUDA-capable GPU to ensure model training can leverage 
    GPU acceleration as indicated by the assertion torch.cuda.is_available().

### ✨Software Dependencies✨

-    **Google Colab**: For mounting Google Drive and notebook operations.
-    **PIL**: For image manipulation.
-    **torch**: Main library for neural network models and GPU computing.
-    **torchvision**: For image transformations and dataset loaders.
-    **pytorch lightning**: For more structured and cleaner training loops.
-    matplotlib: For plotting graphs and visualizations.
-    numpy: For numerical operations.
-    segmentation_models_pytorch: For pretrained segmentation models.
-    plotly: For interactive visualizations.
-    pandas: For data manipulation and reading/writing CSV files.
-    re: For regular expression operations.
-    pathlib: For system path operations.
-    io: For handling IO operations.
-    Other custom scripts (models, dataset_loader, evaluation, util, visualize) are used, 
    so ensure these are included in the system path or installation directory.

### ✨Tested Versions✨

-    The notebook has been configured to work on Google Colab with specific paths
    set for Google Drive. Thus, it's tested on the Google Colab environment with
    internet connectivity and access to Google Drive.

### ✨Instructions✨

-    Open the Jupyter Notebook:
        Open the model_compile.ipynb notebook in Jupyter or Google Colab.
        
-   Run the Notebook:
        Execute the cells sequentially, which will involve mounting the drive, loading data,
        training models, and visualizing results.

### ✨Typical Installation Time✨

-    Local Machine:
    Installation of Python and dependencies should take about 15-30 minutes on a normal
    desktop computer, depending on your internet connection.
    
-    Execution Time:
    The time to run the notebook will heavily depend on the dataset size and your hardware,
    especially whether a GPU is available or not.


[Link to dataset](https://drive.google.com/file/d/1uiP14oo8_4OTx6sBgO-uor0SxDhtsdxG/view?usp=sharing)
