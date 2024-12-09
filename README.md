# ML-creativecoding
Material for the Creative Coding course – Machine Learning Sessions
Link to the presentation: 


## 1. Checking if python is installed

#### MacOS

1.  Open 'Terminal'
2.  type `python3 --version`  

#### Windows

1. Open 'Command Prompt'
2. type `python --version`  

**If you have python 3.9 installed, you can skip step 2**
<br><br><br>

## 2. Installing Python

#### MacOS

1. Install brew with this Terminal Command: 
    ```bash 
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2. Check if brew was installed correctly
    ```bash
    brew --version
    ```

3. Install Python3.9 with brew
    ```bash
    brew install python@3.9
    ```

4. Check if Python3.9 was installed correctly
    ```bash
    python3.9 --version
    ```

5. Set Python3.9 to be the default Python Version
    ```bash
    echo 'alias python3="/usr/local/bin/python3.9"' >> ~/.zshrc
    source ~/.zshrc
    ```

#### Windows

1. Download the Python3.9 installer for Windows(64bit) [here](https://www.python.org/downloads/release/python-3913/) 
2. Double click the downloaded installer
3. In the installation process check "Add Python to PATH"
4. Verify the installation
   ```bash
   python --version
   ```



### Hello World

Open Terminal or Command Prompt and type:

1. ```python3```
2. ```print("Rage Against the Machine Learning")```
3. ```exit()```


<br><br><br>


## 3. Installing YOLO (Mac and Windows)

1. Create a virtual environment
    ```bash
    python3 -m venv Ultralytics
    ````
2. Activate the virtual environment
   ```bash
   source ./Ultralytics/bin/activate
   ```
3. Install PyTorch
   ```
   pip3 install torch torchvision torchaudio
    ```
4. Install Ultralytics
   ```
   pip install ultralytics
   ```
5. Install OpenCV
    ```
    pip install opencv-python
    ```

<br>

**Congrats, installation done!**

<br><br><br>

## 4. Run inference on webcam

1. Create a folder on your machine and give it a name e.g `"YOLOv8_with_Ferdinand"`
2. Open the folder you just created in your favourite code editor
3. Create a new file and call it `webcam.py`

### webcam.py code breakdown:

Import necessary dependencies:
```python
import cv2
from ultralytics import YOLO
```

Specify which YOLO model you want to use. This points to a "weights"-file (.pt) in your folder and can be interchanged with other weights.
```python
model = YOLO('yolov8n.pt')  
```

We want to use our webcam as a source:
```python
cam = cv2.VideoCapture(0) 
```

Read from the Webcam and print an error if the read fails:
```python
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
```

Get results from the YOLO inference:
```python
    results = model(frame)
```

Plot the results so we can see them:
```python
    annotated_frame = results[0].plot()
```

Show the results in a new window:
```python
    cv2.imshow("YoloV8 Webcam", annotated_frame)
```

Break the loop and close the program:
```python
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
````

<br>


<details>
<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

## 5. Finetuning



