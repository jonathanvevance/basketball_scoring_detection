<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/jonathanvevance/basketball_scoring_detection">
    <img src="readme_images/basketball_logo.png" alt="Logo" width="250" height="250">
  </a>

<h2 align="center">Basketball Scoring Detection</h2>

  <p align="center">
    Using Yolov3 and Multi Instance Learning
    <br />
    <a href="https://github.com/jonathanvevance/basketball_scoring_detection"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/irfanthayyil">Irfan Thayyil</a>
    ·
    <a href="https://github.com/Adil-MohammedK">Aadil Muhammed K</a>
    ·
    <a href="https://github.com/akshaykrishh">Akshay Krishna</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This is Team Aai's submission for MTX Shaastra 2022 Hackathon for problem statement #1. We have used a <a href = "https://arxiv.org/abs/1804.02767"> yolov3 objection detection model </a> for detecting basketball hoops in each frame, and after cropping out the basket, we run a simple convolutional network which is trained using <a href = "https://en.wikipedia.org/wiki/Multiple_instance_learning"> multi instance learning </a>.

<p align="right">(<a href="#top">back to top</a>)</p>


### Built With

* [Pytorch](https://pytorch.org//)
* [React.js](https://reactjs.org/)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

These instructions are written for an Ubuntu system. If you have a Windows system, please refer to <a href="https://docs.microsoft.com/en-us/windows/wsl/about"> WSL on Windows</a>.

### Prerequisites
It is required to bypass password prompts on Ubuntu so that all functions work properly. To do this, you may refer to <a href="https://phpraxis.wordpress.com/2016/09/27/enable-sudo-without-password-in-ubuntudebian/"> this link</a>.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/jonathanvevance/basketall_scoring_detection.git
   ```
2. From the root directory,
   ```sh
   sudo bash setup.sh
   ```
3. From the root directory,
   ```sh
   cd src/yolov3_helper;
   sudo bash setup.sh
   ```
4. To prevent file permission issues, from the root directory,
   ```sh
   sudo chmod -R 777 data
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Running the web app

### Evaluating the model on some dataset:
1. Place the dataset in data/evaluation/eval_dataset folder. The folder structure expected is:

        eval_dataset
            ├── scoring_clips
                ├── clip_1.mp4
                ├── clip_2.mp4
                ├── ...
            ├──non_scoring_clips
                ├── clip_4.mp4
                ├── clip_6.mp4
                ├── ...

2. 2. From the root directory, activate the python venv by running:
    ```sh
    source .env/bin/activate
    ```
3. Edit the src/configs/eval_config.py file with the required evaluation settings.
4. From the root directory, start training by running:
    ```sh
    python src/evaluate.py
    ```

### Training the model on some dataset:

1. Place the dataset in data/training/hackolympics_data folder. The folder structure expected is:

        hackolympics_data
            ├── scoring_clips
                ├── clip_1.mp4
                ├── clip_2.mp4
                ├── ...
            ├──non_scoring_clips
                ├── clip_4.mp4
                ├── clip_6.mp4
                ├── ...

2. From the root directory, activate the python venv by running:
    ```sh
    source .env/bin/activate
    ```
3. From the root directory, prepare the dataset by running:
    ```sh
    python src/make_dataset.py --videos --yolov3 --split
    ```

4. Edit the src/configs/train_config.py file with the required training settings.
5. From the root directory, start training by running:
    ```sh
    python src/train.py
    ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [] Interactive video player in the web application. 
- [] Inference speedup using TensorRT on GPU and Intel OpenVino on Intel CPU.

See the [open issues](https://github.com/jonathanvevance/basketall_scoring_detection/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact Us

<!-- Jonathan Ve Vance - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com -->
1. Jonathan Ve Vance - [Linkedin](https://linkedin.com/in/jonathanvevance) - jonathanvevance@gmail.com
2. Irfan Thayyil
3. Aadil
4. Akshay Krishna


Project Link: [https://github.com/jonathanvevance/basketall_scoring_detection](https://github.com/jonathanvevance/basketall_scoring_detection)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []() We obtained pretrained weights for basket (hoop) detector yolov3 model from <a href = "https://github.com/SkalskiP/ILearnDeepLearning.py"> this great repository</a>. Huge shoutout to the author.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: images/screenshot.png