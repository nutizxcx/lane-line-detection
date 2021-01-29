# Lane Line Detection
<em>CPE463 Image Processing project, May 2020</em>

**Editors**
<ol>
<li>60070501036 Pasit Hankijpongpan</li>
<li>60070501009 Chanakarn Sapdeemeecharoen</li>
<li>61070505202 Kittapat Ratanaphupha</li>
</ol>

## Overview project
![Overall process](https://github.com/nutizxcx/lane-line-detection/blob/master/pictures/overall.jpg)
The objective of the project is for
> Identify straight lane lines from a variety of images recorded from video, which is the beltway road in daytime.
### Possible solution
For the detail, the proposed solution is following as below
1. Convert original image to grayscale.
2. Darkened the grayscale image (this help in reducing contrast from discoloured regions of road)
3. Convert original image to HLS colour space.
4. Isolate yellow from HLS to get yellow mask. ( for yellow lane markings)
5. Isolate white from HLS to get white mask. (for white lane markings)
6. Bit-wise OR yellow and white masks to get common mask.
7. Bit-wise AND mask with darkened image .
8. Apply slight Gaussian Blur.
9. Apply canny Edge Detector (adjust the thresholds — trial and error) to get edges.
10. Define Region of Interest. This helps in weeding out unwanted edges detected by canny edge detector.
11. Retrieve Hough lines.
12. Consolidate and extrapolate the Hough lines and draw them on original image.

## Data Sources
Image of beltway road in daytime with clear weather.
https://github.com/kenshiro-o/CarND-LaneLines-P1
