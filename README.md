# Scuba-diving-gesture-recognition
Scuba diving gestures recognition using Mediapipe, cv2 and PyTorch

## Inspiration
I was always been a huge fan of Minority Report and awaited the day when we could use gestures for our day to day use. Then came [Pranav Mistry with his Sixth Sense technology](https://www.ted.com/talks/pranav_mistry_the_thrilling_potential_of_sixthsense_technology) which blew my mind. However it was too hardware focussed.

Google came out with [Mediapipe](https://research.google/pubs/pub48292/) in 2019. I had just completed my Scuba certifications in Open and Advanced Open Water scuba diving when I came across some cool animations on Facebook using Mediapipe. A quick search led me to [Nicholas Renotte's](https://www.linkedin.com/in/nicholasrenotte/) famous Sign Language [video](https://www.youtube.com/watch?v=doDUihpj6ro). I was super impressed by the process of webcam images using cv2 and implemented the approach for scuba diving signals (I had just completed by Open Water and Advanced Open water courses then) using Pytorch.

## Mediapipe:
### Paper - [MediaPipe: A Framework for Building Perception Pipelines](https://arxiv.org/abs/1906.08172)
### Repo - [Github repo](https://github.com/google/mediapipe)
### Documentation - https://google.github.io/mediapipe/



## Key learnings:
<ul>
<li> Mediapipe landmarks can get significantly impacted by the lighting - both at the time of data collection and at the time of inference. </li>
<li> If you're not careful while consolidating the various frames for your input dataset, the order of labels can get scattered. After completing your one-hot encoding, run a sample check across your classes to determine its index in the encoding </li>
<li> More samples ! I could record only 150 samples across 5 different action classes </li>
<li> Stability over precision. Video processing has this annoying property of rapidly changing classes as frames change. So to avoid this, it takes this model about 3-4 frames before it stabilizes its prediction class. Hence, you may see a bit of jitter in the displayed result but it will immediately stabilize. I had this issue in my [object classification project](https://github.com/SwamiKannan/Formula1-car-detection)


## Sample video
![sample_video](scuba_diving.gif)

 