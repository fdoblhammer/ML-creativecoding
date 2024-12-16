# ML-creativecoding
Material for the Creative Coding course – Machine Learning Sessions

Link to the presentation: https://www.figma.com/deck/Bbpi7aXt2P9arA8wzX4fnU/doblhammer.media-Pr%C3%A4sentation?node-id=75-423&node-type=slide&viewport=-153%2C-93%2C0.71&t=kSRoTcAWWosUiQrs-1&scaling=min-zoom&content-scaling=fixed&page-id=0%3A1

<br><br><br>

## ML in Projects

Dries Depoorter – [Recharge](https://driesdepoorter.be/recharge/)

Dries Depoorter – [Border Birds](https://driesdepoorter.be/borderbirds/)

Trevor Paglen – [ImageNet Roulette](https://paglen.studio/2020/04/29/imagenet-roulette/)

Kate Crawford & Trevor Paglen – [Article "Excavating AI"](https://excavating.ai/)

Heather Dewey-Hagborg – [Gram's Faces](https://deweyhagborg.com/projects/gram-s-faces)

Adam Harvey - [CV Dazzle](https://adam.harvey.studio/cvdazzle/)

Shinseungback Kimyonghun – [Cloud Face](https://ssbkyh.com/works/cloud_face/)

Total View – [Sensing Sinicization](https://doblhammer.media/project/sensing-sinicization)

Ferdinand Doblhammer – [Sibling Inference](https://doblhammer.media/project/sibling-inference)

Dries Depoorter – [The Flamish Scrollers](https://driesdepoorter.be/theflemishscrollers/)

<br><br><br>


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
   on Windows
   ```bash
   .\Ultralytics\Scripts\activate
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
model = YOLO('yolo11n.pt')  
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

**Confidence Threshold**

You can set the confidence threshold to a value between 0 and 1. Detections below will not be shown. Change this line:

```python
results = model(frame, conf=0.5)
```

**IOU Threshold**

Specifies how much the detections can overlap – this is used to eliminate multiple detections on the same object to get a clear output
```python
results = model(frame, iou=0.5)
```

**Image Size**

Sets how the size of the image that will be shown to the detector. Larger images result in slower detections.
```python
results = model(frame, imgsz=1280)
```


**Device**

If you have a graphics card (NVIDIA or Apple M1/2/3/4 Chip) you can run the inference on it. This drastically improves the speed.
```python
#CPU
results = model(frame, device=cpu)
#GPU
results = model(frame, device=0)
```

**Maximum Number of Detections**

Sets the max number of detection. Useful if you just want to detect 1 object.
```python
results = model(frame, max_det=1)
```

 

**Classes**

Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
```python
results = model(frame, classes=[0])
```

For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

## 6. Using different datasets

To begin, lets try out some other YOLO models from Ultralytics. They will be downloaded automatically when you start the program.

### Official YOLO-Weights from Ultralytics

**Segmentation**
```python
model = YOLO('yolo11n-seg.pt')
```

**Pose Estimation**
```python
model = YOLO('yolo11n-pose.pt')
```

**Classification**
```python
model = YOLO('yolo11n-cls.pt')
```

### 
