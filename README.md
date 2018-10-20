# Sudoku Solver

A simple Sudoku solver made in python 3.6
**Note:** Try to keep the source, i.e the paper flat. Else, the OCR might not work.


## Running
For simplicity, you can download the executable from the Google Drive link in `drive.txt`
**NOTE:** This only works on Windows. For that, you need to have Tesseract Installed. **The application doesn't work without Tesseract**
Install Tesseract at 

> C:/Program Files (x86)/Tesseract-OCR

[Tesseract (Source Forge)](https://sourceforge.net/projects/tesseract-ocr/)

The source code is in `master_sudoku.py`

## Dependencies

The following dependencies are required:

    numpy
    opencv
    pygame
    pytesseract
Alternatively, you can run:

    pip install -r requirements.txt
Also, you should download the Tesseract OCR. Here is the link for it:
[Tesseract (Source Forge)](https://sourceforge.net/projects/tesseract-ocr/)


If you are using the souce code, you should update the path of the Tesseract OCR in the code. I have commented the instructions for further details.

## Additional Information

I have used Tesseract OCR for image recognition. If you want, you can train a custom Tensorflow Model and import it for character recognition. The `predict_2.py` file is one such model. I have not used it since the recognition speed is low. But it has a higher accuracy.

The model requires the `model2.ckpt` file to work.
For generating the `model2.ckpt` file, run the training script `create_model_2.py` by using the command

    python create_model_2.py

But first, take the following files:

    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
and put them into a folder called `MNIST_data` and then run the script. Else, you can change the path of the files in the `create_model_2.py`

## Credits
This project was developed by Rochana HM.
