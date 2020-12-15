# Room-Occupancy-Detection-using-YOLOv4
This uses YOLOv4 to detect occupancy in a room and counts the number of people.  

## Dependencies-
* opencv-python 4.4.0.46
* numpy
* PahoMQTT
* SheetsAPI

## Main logic behind occupancy detection
The Centroid Position of people's bounding boxes are tracked and the following conditions are applied for occupancy detection-

![Logic Behind occupancy detection](https://github.com/sakshamprakash01/Room-Occupancy-Detection-using-YOLOv4/blob/main/occupancy.png?raw=true)

## Video Demonstration

[![Watch the video here!](https://img.youtube.com/vi/c89ByrbQ5dA/maxresdefault.jpg)](https://www.youtube.com/watch?v=c89ByrbQ5dA)
