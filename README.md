# Scuba-diving-gesture-recognition
Scuba diving gestures recognition using Mediapipe, cv2 and PyTorch

## Inspiration
I was always been a huge fan of Minority Report and awaited the day when we could use gestures for our day to day use. Then came [Pranav Mistry with his Sixth Sense technology](https://www.ted.com/talks/pranav_mistry_the_thrilling_potential_of_sixthsense_technology) which blew my mind. However it was too hardware focussed.

Google came out with [Mediapipe](https://research.google/pubs/pub48292/) in 2019. I had just completed my Scuba certifications in Open and Advanced Open Water scuba diving when I came across some cool animations on Facebook using Mediapipe. A quick search led me to [Nicholas Renotte's](https://www.linkedin.com/in/nicholasrenotte/) famous Sign Language [video](https://www.youtube.com/watch?v=doDUihpj6ro). I was super impressed by the process of webcam images using cv2 and implemented the approach for scuba diving signals (I had just completed by Open Water and Advanced Open water courses then) using Pytorch.

## Mediapipe:
### Paper - [MediaPipe: A Framework for Building Perception Pipelines](https://arxiv.org/abs/1906.08172)
### Repo - [Github repo](https://github.com/google/mediapipe)
### Documentation - https://google.github.io/mediapipe/
<br>


## Contents:
<b>I. [Data capture](https://github.com/SwamiKannan/Scuba-diving-gesture-recognition/blob/main/I.%20Data%20capture.ipynb)</b>
<ul><li>Testing the camera, mediapipe library (To ensure adequacy of lighting / setup, get fps of the camera to calculate sequence length</li>
<li> Capture data from the webcam for various actions i.e. for 5 actions, gather 20 samples, each of which is a 1 second video (30 frames)</li></ul>

<b>II. [Data Processing](https://github.com/SwamiKannan/Scuba-diving-gesture-recognition/blob/main/II.%20Data%20processing.ipynb) </b>
<ul><li> Convert 1500 numpy files of gestures,each of 63 points, to a 150X30X63 tensor matrix</li>
<li>One-hot code labels and save both files</li>
<liCreate train and test splits</li></ul>

<b>III. [Model training](https://github.com/SwamiKannan/Scuba-diving-gesture-recognition/blob/main/III.%20Training.ipynb)</b>
<ul><li>Import model architecture and train the model on a single batch of 142 samples (after train and test split)</li>
<li>Train on test clips</li></ul>

<b>IV. [Testing on live data](https://github.com/SwamiKannan/Scuba-diving-gesture-recognition/blob/main/IV.%20Testing%20on%20real-time%20data.ipynb)</b>
<ul><li>Testing on live data feed</li>
<li>Processing and storing the renders</li></ul>

## Key learnings:
<ul>
<li> Mediapipe landmarks can get significantly impacted by the lighting - both at the time of data collection and at the time of inference. </li>
<li> If you're not careful while consolidating the various frames for your input dataset, the order of labels can get scattered. After completing your one-hot encoding, run a sample check across your classes to determine its index in the encoding </li>
<li> More samples ! I could record only 150 samples across 5 different action classes </li>
<li> Stability over precision. Video processing has this annoying property of rapidly changing classes as frames change. So to avoid this, it takes this model about 3-4 frames before it stabilizes its prediction class. Hence, you may see a bit of jitter in the displayed result but it will immediately stabilize. I had this issue in my [object classification project](https://github.com/SwamiKannan/Formula1-car-detection)</li></ul>


## Sample video
<p align="center">
<img src="scuba_diving.gif"><br>
 <sub>Gestures will take a second to align to the correct label</sub>
</p>

 
