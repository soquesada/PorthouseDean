# PorthouseDean

## User Interface

The application can be launched using the executable file, `Difference Detector.exe`. This is the file that can be distributed for use and can be found through a OneDrive link via the `README.md` file in the sub-directory `PorthouseDean\dist\`.

A screenshot of the user interface can be seen below.

![Screenshot of the user interface at first launch](readme_assets\user_interface.png)

Users must select the option “Get comparison image” which will launch a file dialog box to select a file. This file must contain two images side by side wherein the left is the “before” image and the right is the “after” image. Results can take a while to load.

After it has loaded, use the mouse scroll wheel to navigate the large box to zoom in and out of the images produced. Users can also save the resulting image locally using the navigation bar at the top and clicking the tab that says “Save file”.

The Python script `MainWindow.py` can be used as an alternative way to launch the UI. Functions in the `ImageDifference.py` script is used in the former file to carry out the data preparation and highlighting of the changes. Ensure your current directory is the directory where the Python script is located and execute either of these commands:

```
python MainWindow.py
python -u "path\to\file\MainWindow.py"
```

An example of the results produced after inputting an image can be seen below.

![Example of the differences highlighted in the image](readme_assets\result1.png)

## Machine Learning Scripts

The folders and files I used for the classification of floor plans can be found in the sub-directory `PorthouseDean\ml_scripts\`. Due to the .zip files being too large to push to the repository, I have linked those in a separate OneDrive link in the `README.md` file in that directory. All information as to the tutorials, videos, and any other external resources I used can also be read in the markdown file.

## Documentation

The documentation for the project is scattered throughout the `README.md` files in the appropriate directories. The overall report produced is the file in the `Internship Report.pdf`.