# IMU classifier using deep learning with low memory usage (STM32 compatible)

IMU classifiers can have a great business value in various industries. For example, you could train a neural network to classify if the human is walking, sitting, jumping, etc... Or perhaps we could train to classify if some kind of arbitrary robot is driving, standing still, driving with some tools on, or standing with some tools on. This repository provides the sample dataset, trained model, and training pipeline for this classification problem. 

A PyTorch model is designed, built, and trained with the data from the gyroscope and accelerometer. The model later is then saved as onnx model and later used on stm32L433, where it is uploaded to the microcontroller using STM32CubeMX with the help of X-CUBE-AI expansion. 

## Dependencies

- PyTorch,
- TensorBoard,
- NumPy,
- PIL,
- TorchMetrics,
- onnxruntime
- h5py

## Collected data

Folder data_raw contains the raw dataset collected from our IMU. It is in a CSV format, where columns are sensor axis and each subsequent row is a sensor reading. 

![Untitled Diagram](https://user-images.githubusercontent.com/53571191/159280351-0630f221-7a92-4ec5-be9d-e8927ceb6226.jpg)

## Hyper parameters

The main.py in the root of the project contains a parser for main.py arguments. 
![image](https://user-images.githubusercontent.com/53571191/159281647-befa98ab-b003-4d40-b048-d1cb9018547c.png)

The optimizer and loss function naming should follow the pascal case. You can find the usable optimization functions and loss functions here https://pytorch.org/docs/stable/nn.html. 

## Running the model

To start training simply run main.py in the root folder of the project.

To see the accuracy and confusion matrix open new terminal. Write:
tensorboard --logdir="results\robot_state_classification"

![image](https://user-images.githubusercontent.com/53571191/159283375-a619ab2a-09ca-4e0b-aa5a-d722cb936d76.png)

![image](https://user-images.githubusercontent.com/53571191/159283437-152f5271-a2fd-42c9-8f30-fe4455e255c7.png)

CNN stats:

Parameter count: 4456
Parameter dtype: float32
Parameter size: 17.40625KB
Size of Parameters + input: 22.09375KB

Highest achieved accuracy for our classification problem: 99%
