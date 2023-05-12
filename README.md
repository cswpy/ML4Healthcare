# ML4Healthcare


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!--
 *** <a href="https://github.com/cswpy/ML4Healthcare">
 ***   <img src="images/logo.png" alt="Logo" width="80" height="80">
 *** </a>
-->
<h3 align="center">Breast Cancer Stage Classification</h3>

  <p align="center">
    Line Following Robot implementation on Jetbot
    <br />
    <a href="https://app.nightingalescience.org/contests/vd8g98zv9w0p"><strong>Explore the Competition »</strong></a>
    <br />
    <br />
    <a href="https://github.com/cswpy/Ml4Healthcare/issues">Report Bug</a>
    ·
    <a href="https://github.com/cswpy/Ml4Healthcare/issues">Request Feature</a>
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains code for a road-following car built using the JetBot platform. The JetBot is a small, AI-enabled robot that is designed to teach machine learning concepts and robotics principles. This car uses image processing to detect black lines and green boxes to follow a track. The green boxes indicate U-turns and other turns. More detail about the task tackled can be found <a href="https://github.com/BaraaAlJorf/Jetbot_Linefollowing/tree/main/Detailed_Report">here</a>

### Built With

* [![Jupyter][Jupyter.com]][Jupyter-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Prerequisites

To run this code, you will need a JetBot with a camera and a working installation of the JetPack software. You will also need the following Python libraries:

-traitlets
-ipywidgets
-cv2 (OpenCV)
-numpy
-PIL (Python Imaging Library)

### Jetbot Setup

The JetBot is an open-source AI robot platform based on the NVIDIA Jetson Nano that is designed to be easy to build, assemble, and use. Here are the steps to build and set up a JetBot:

1. Obtain the necessary hardware components: You will need a Jetson Nano Developer Kit which comes with a camera, motors, wheels, battery, and other necessary components 
2. Follow the steps on https://jetbot.org/master/index.html to setup the appropriate OS on your Jetbot

### Installation

Clone the repo onto your jetbot
   ```sh
   git clone https://github.com/BaraaAlJorf/Jetbot_Linefollowing.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

To use this code, simply clone the repository onto your JetBot and run the 'escape-2.ipynb' notebook. This will start the camera stream and begin processing images to follow the black lines.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Code Overview
The road_following.ipynb notebook contains the code for the road-following car. Here is a brief overview of the main components:

#### Camera and Robot Instances
The code creates instances of the Camera and Robot classes provided by the JetBot library. These instances are used to capture images and control the movement of the car.

#### Image Processing
The code processes the camera images to detect black lines and green boxes. It does this by:

Identifying black and green pixels in the image.
Preprocessing the image by removing noise using erode and dilate operations.
Finding contours in the image.
Drawing contours onto the image for debugging purposes.
Calculating the angle and center of the black boxes.
Sending the "error" distance from center and angle to the PID controller.
#### Main Loop
The main loop of the code processes images captured by the camera in real time. It isolates the black line in the image, removes noise, finds contours, and calculates the angle and center of the black boxes. It then sends this information to the PID controller to adjust the movement of the car.

If the black line is not detected for a certain number of frames, the car will stop moving and the program will exit.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions or suggestions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Baraa Al Jorf - baj321@nyu.edu, Manuel Padilla -map971@nyu.edu

Project Link: [https://github.com/BaraaAlJorf/Jetbot_Linefollowing](https://github.com/BaraaAlJorf/Jetbot_Linefollowing)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-url]: https://github.com/BaraaAlJorf/Jetbot_Linefollowing/graphs/contributors
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[forks-shield]: https://img.shields.io/github/forks/BaraaaALJorf/Jetbot_Linefollowing.svg?style=for-the-badge
[forks-url]: https://github.com/BaraaAlJorf/Jetbot_Linefollowing//network/members

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/baraaaljorf/

[Jupyter.com]: https://jupyter.org/assets/homepage/main-logo.svg
[Jupyter-url]: https://jupyter.org/
