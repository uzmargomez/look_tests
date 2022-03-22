# Look Tests

## Introduction 
This is a full face detection and recognition system. For the detection part [MTCNN](https://github.com/ipazc/mtcnn) package is used. The recognition is achieved by using both a sparse representation method and using a Keras implementation of [FaceNet](https://github.com/faustomorales/keras-facenet) to get the embeddings.

## Getting Started

### Sparse Representation.

1. Clone this repository. After that, you need to unzip the example dataset contained in face_recognition_sparseRep/datasets.

2. This code runs with Python, you need to execute the following command inside the "face_recognition_sparseRep" directory if you are using pip (or the analogous if using conda)

    ```
    pip3 install -r requirements.txt
    ```

    preferably using a virtual environment. 

3. a) Run the following command in order to start the service to which the webcam will send the images to be classified

    ```
    python3 server.py
    ```

    b) Run the following command in order to start the webcam

    ```
    python3 webcam.py
    ```

4. To check the classification for the test set, simply run the command

    ```
    python3 testing_accuracy.py
    ```

    Inside the file there is a description of the parameters that can be change for the testing study. 

5. The plots are generated using the results of the testing_accuracy.py code. The notebook is [here](face_recognition_sparseRep/notebooks/studies_classification.ipynb)

### Embeddings

1. Clone this repository. After that, you need to unzip the example dataset contained in face_recognition_embeddings/datasets.

2. With Docker installed, run the following command inside the "face_recognition_embeddings" directory

    ```
    docker build -t look_test:v1 .
    ```

3. After the image is created, run the following command

    ```
    docker run --gpus all -p 8501:8501 look_test:v1
    ```

4. Open the app in [http://localhost:8501/](http://localhost:8501/)