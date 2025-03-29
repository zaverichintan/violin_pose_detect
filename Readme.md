The project has two files, task_a.py for Task 3 (subtask a) and task_b_mediapipe_angle.py for Task 3 (subtask b).

To create virtualenv 
```bash 
python -m venv venv
./venv/bin/activate 
pip install -r requirements.txt
```


## Task a 
For task_a, the path of input video file can be changed in the main function variable VIDEO_PATH or pass it to the argument. 
It uses a onnx model converted from mediapipe tensorflow model which gives 17 joints 2D location along with confidence. 
There is also angle calculation logic which calculates 2D angle between the shoulder, elbow and wrist. 

To run the code:
```bash 
python task_a.py data/ed_sheeran_violin_cover.mp4
```

or pass 0 to read from webcam
```bash 
python task_a.py 0 
```

## Task b 
For task_b, the video is being read from webcam and the Blaze pose model is used to get X, Y and Z coordinates. 
The angles are not very accurate as the Z coordinate estimation is not very acccurate 
 
To run the code: 
```python task_b_mediapipe_angle.py```

