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
    <a href="https://github.com/Adil-MohammedK">Adil Muhammed K</a>
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

[![Product Name Screen Shot][product-screenshot]](#about-the-project)

This is Team Aai's submission for MTX Shaastra 2022 Hackathon for problem statement #1. We have used a <a href = "https://arxiv.org/abs/1804.02767"> yolov3 objection detection model </a> for detecting basketball hoops in each frame, and after cropping out the basket, we run a simple convolutional network which is trained using <a href = "https://en.wikipedia.org/wiki/Multiple_instance_learning"> multi instance learning </a>.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

- [Pytorch](https://pytorch.org//)
- [React.js](https://reactjs.org/)
- [NodeJS](https://nodejs.org/en/)
- [Docker](https://www.docker.com/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

These instructions are written for an Ubuntu system. If you have a Windows system, please refer to <a href="https://docs.microsoft.com/en-us/windows/wsl/about"> WSL on Windows</a>.

### Prerequisites

It is required to bypass password prompts on Ubuntu so that all functions work properly. To do this, you may refer to <a href="https://phpraxis.wordpress.com/2016/09/27/enable-sudo-without-password-in-ubuntudebian/"> this link</a>.

### Docker Usage

#### Building image from Dockerfile

(This method uses less data of about 1.6 GB but will need to build from scratch)

1. Clone the repo.
   ```sh
   git clone https://github.com/jonathanvevance/basketball_scoring_detection.git
   ```
2. Change to docker branch
   ```sh
   git checkout docker
   ```
3. From root of repo, Build docker image
   ```sh
   docker build . -t aai-mtx
   ```
4. Run docker container
   ```sh
   docker run --gpus all -p 3000:3000 -p 4000:4000 -p 5000:5000 --name basketball aai-mtx
   ```
5. Load the UI by using the [link](http://localhost:3000)
6. To check logs of docker container,
   ```sh
   docker logs basketball
   ```
7. To stop docker container,
   ```sh
   docker stop basketball
   ```
8. To remove container,
   ```sh
   docker rm basketball
   ```

#### Pulling from Docker Hub

(Warning: This image is about 5.6 GB. Use this method if you have fast network and enough data)

1. Pull docker image.
   ```sh
   docker pull adilmohammed2000/aai-mtx
   ```
2. Run docker container
   ```sh
   docker run --gpus all -p 3000:3000 -p 4000:4000 -p 5000:5000 --name basketball adilmohammed2000/aai-mtx
   ```
3. Load the UI by using the [link](http://localhost:3000)
4. To check logs of docker container,
   ```sh
   docker logs basketball
   ```
5. To stop docker container,
   ```sh
   docker stop basketball
   ```
6. To remove container,
   ```sh
   docker rm basketball
   ```

#### Possible error in Docker

1. docker: Error response from daemon: Ports are not available: listen tcp 0.0.0.0:3000: bind: An attempt was made to access a socket in a way forbidden by its access permissions.  
   Try this solution: [Solution](https://stackoverflow.com/questions/57891647/port-issue-with-docker-for-windows/66865808#66865808) or close any programs using these ports: 3000,4000,5000

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
   cd src/yolov3_helper
   sudo bash setup.sh
   ```
4. To prevent file permission issues, from the root directory,
   ```sh
   sudo chmod -R 777 data
   ```
5. To install NodeJS backend dependencies, from the root directory. **If npm and node not installed in your system**, refer the section below
   ```sh
   cd src/server/
   npm install
   npm install -g serve
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Possible Errors

1. /usr/bin/env: ‘bash\r’: No such file or directory (in WSL). To solve,

   ```sh
   sudo nano /etc/wsl.conf
   ```

   Add following line to file,

   ```sh
   [interop]
   appendWindowsPath = false
   ```

   Then restart WSL with `wsl --shutdown` in PowerShell.

2. npm and node not installed in Fresh WSL or Ubuntu installation. To install npm and node,
   ```sh
   sudo apt-get install curl
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   ```
   To verify installation, enter: `command -v nvm` .This should return 'nvm', if you receive 'command not found' or no response at all, close your current terminal, reopen it, and try again.  
   Then to install nodeJS,
   ```sh
   nvm install --lts
   ```
   Verify that Node.js is installed and the currently default version with: `node --version`. Then verify that you have npm as well, with: `npm --version`

<!-- USAGE EXAMPLES -->

## Usage

### Running the web app

1. Go to root of repo folder
2. In one terminal, from root directory, start Python server by
   ```sh
   source .env/bin/activate
   python3 src/server.py
   ```
3. In second terminal, from root directory, start the backend nodeJS server,
   ```sh
   node src/server/server.js
   ```
4. In third Terminal, from root directory, start the frontend UI by,
   ```sh
   serve -s src/build
   ```
   Go to link shown in the terminal. Or open a browser and use this [link](http://localhost:3000)

### Evaluating the model on some dataset:

1.  Place the dataset in data/evaluation/eval_dataset folder. The folder structure expected is:

        eval_dataset
            ├── scoring_clips
                ├── clip_1.mp4
                ├── clip_2.mp4
                ├── ...
            ├── non_scoring_clips
                ├── clip_4.mp4
                ├── clip_6.mp4
                ├── ...

2.  2. From the root directory, activate the python venv by running:
    ```sh
    source .env/bin/activate
    ```
3.  Edit the src/configs/eval_config.py file with the required evaluation settings.
4.  From the root directory, start training by running:
    ```sh
    python src/evaluate.py
    ```

### Training the model on some dataset:

1.  Place the dataset in data/training/hackolympics_data folder. The folder structure expected is:

        hackolympics_data
            ├── scoring_clips
                ├── clip_1.mp4
                ├── clip_2.mp4
                ├── ...
            ├── non_scoring_clips
                ├── clip_4.mp4
                ├── clip_6.mp4
                ├── ...

2.  From the root directory, activate the python venv by running:
    ```sh
    source .env/bin/activate
    ```
3.  From the root directory, prepare the dataset by running:

    ```sh
    python src/make_dataset.py --videos --yolov3 --split
    ```

4.  Edit the src/configs/train_config.py file with the required training settings.
5.  From the root directory, start training by running:
    ```sh
    python src/train.py
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [] Interactive video player with seek bar in the web application.
- [] Inference speedup using TensorRT on GPU and Intel OpenVino on Intel CPU.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact Us

<!-- Jonathan Ve Vance - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com -->

1. Jonathan Ve Vance - [Linkedin](https://linkedin.com/in/jonathanvevance) - jonathanvevance@gmail.com
2. Irfan Thayyil - [LinkedIn](https://www.linkedin.com/in/mohammed-irfan-thayyil-34311a166) -irfanthayyil@gmail.com
3. Adil Muhammed K - [LinkedIn](https://www.linkedin.com/in/adil-mohammed-065603155) - adilmohammed2000@outlook.com
4. Akshay Krishna - [LinkedIn](https://www.linkedin.com/in/akshaykrishh/) - akshaykrishnakanth@gmail.com

Project Link: [https://github.com/jonathanvevance/basketall_scoring_detection](https://github.com/jonathanvevance/basketall_scoring_detection)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- []() We obtained pretrained weights for basket (hoop) detector yolov3 model from <a href = "https://github.com/SkalskiP/ILearnDeepLearning.py"> this great repository</a>. Huge shoutout to the author.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[product-screenshot]: readme_images/app_screenshot.png
