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


