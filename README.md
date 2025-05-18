# DJI/Ryze Tello Computer Vision

It follows whatever aruco tag it sees, that's mostly it right now.
To calibrate, print out an openCV checkerboard, and run main.py and click E while holding the checkerboard in front of the drone's camera (theres a viewfinder), and make sure to get about 20 good pictures. Then you can run calibrate.py and after a few minutes, you're good to print out a tag of your own (image is in the repo). Scale it to 200% on letter paper to get good distances. If you change the size, configue the `aruco_marker_side_length` variable to get the correct distances.

## Controls

There is limited controls built in, with plans to make the better in the future.

SPACE - Take Off

W - Forwards 1 step

S - Backwards 1 step

A - Rotate Left

D - Rotate Right
