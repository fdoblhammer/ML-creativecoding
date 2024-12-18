# ML-creativecoding Session 02

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


## 1. Restarting YOLO11

1. Open your dedicated YOLO Folder on your Machine with Terminal/Command Prompt/Powershell

    *Either* – cd into your folder. You can type `cd + SPACE` and drag your folder into the Terminal/Command Prompt Window
    ```bash
    cd path/to/your/folder
    ```

    *Or* – Right-click on your folder and select `New Terminal at folder...`/`Open in Terminal`

2. Activate your virtual environment (skip if you're not using it)

    **MacOS**
    ```bash
    source ./Ultralytics/bin/activate
    ```

    **Windows**
    ```bash
    .\Ultralytics\Scripts\activate
    ```

3. Run your YOLO11 script

    ```bash
    python webcam.py
    ````


**Tips**

To stop the virtual environment just type `deactivate`

<br><br><br>

## 2. Finetuning YOLO

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

<br><br><br>

## 3. The Flamish Scrollers

A simplified reconstruction of Dries Depoorters work. We'll modify our Python script to print a message when both the classes `person` and `cell phone` are being detected.

We will modify our webcam script, if you need the code its here:
<details>
<summary>Code Webcam.py</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

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

    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

1. **Find out the number of the classes we want to detect**

    For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

    <details>

    <summary>Classes</summary>

    `person: 0`
    `cell phone: 67`

    </details>


2. **Set the detector to only detect these two classes**

    Change this line:

    ```python
    results = model(frame, classes=[0, 67])
    ```

3. **Unclutter your print**

    The YOLO detector should now only detect the set two classes. Now we want to get a message in the print if both classes are seen. But YOLO is already printing lots of messages – let's suppress those first

    Change this line:

    ```python
    results = model(frame, classes=[0, 67], verbose=False)
    ```

4. **Check if results are coming in and read from the detection box**



    after the line with `results` add this:

    ```python
    detected_classes = set()
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            cls = int(box[5]) 
            detected_classes.add(cls)
    ```

    This creates the variable `detected_classes`. If results from the detector are coming in, we look for the fifth value of the detection box – which is the class number – and stores it to `detected_classes`

    If you want to know how a for loop works check [this](https://www.w3schools.com/python/python_for_loops.asp) for reference.

5. **Print if both classes are detected**

    This condition triggers the print if both classes `0` and `67` are detected.

    ```python
    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    ```
<br>

<details>

<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO('yolo11n.pt')  

confidence_threshold = 0.1

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame, conf=confidence_threshold, classes=[0, 67])

    detected_classes = set()
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            cls = int(box[5]) 
            detected_classes.add(cls)

    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>


<br><br><br>

## 4. A folder of images

Configure YOLO to run the inference on a set of images.

1. Single image:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image_file = "images/original_22b598409ede56d4b39e194cad83b495.jpg"

results = model(image_file, save=True, conf=0.25)

print("done")
```

<br>

2. Images in folders:

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt") 

image_dir = Path("images/")

image_files = [str(p) for p in image_dir.glob("*") if p.is_file()]

# -- Uncomment the next line if you also want to search in subfolders
#image_files = [str(p) for p in root_dir.rglob("*") if p.is_file()]

for image_file in image_files:
    print(f"Processing {image_file}")
    results = model(image_file, save_crop=False, save=True, save_txt=False, conf=0.4)
```

<br><br><br>

## 5. Video

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
```

<br><br><br>


## 6. Custom Training

#### What does a training set look like?
- Consists of a lot of representative images of the objects/things the algorithm should detect.
- Normally, results start to get good at 800+ images per class (=type of object)
- For each image, there is a corresponing 'label' file, which holds the information on the class (=what object) and its position on the image (=coordinates)
- The structure of a YOLO Dataset typically looks like this:
- - A folder `train` containing 80% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt)
- - A folder `val` containing 20% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt) 
- - A `.yaml`file containing information about our classes and folder location

<br><br><br>

#### Download an annotated Training Dataset and train it on your machine

Sources:

[Roboflow](universe.roboflow.com)
[kaggle](kaggle.com)

A not very sophisticated dataset:
[Open/Close Eyes Dataset](https://universe.roboflow.com/isee-gufmk/eyes-zefum/dataset/6)

<br>

1. Extract downloaded dataset folder into your project folder.
2. In your project folder create a file named `train.py`
3. Code:
    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="yourdatasetfolder/data.yaml", epochs=100, imgsz=640)
    ```
4. Start the training
    ```bash
    python train.py
    ```

    You will probably encounter an error when running this because 

5. Wait until finished
6. Navigate to the newly created folder `runs/train/weights`and find `best.pt`
7. Copy best.pt save it to a different location and name it `mytraining.pt`

<br><br><br>

#### Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.


