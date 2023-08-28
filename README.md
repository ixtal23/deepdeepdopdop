# Deep Deep Dop Dop

This is a deepfake tool that implements the swapping and restoration of faces in images and videos by [InsightFace](https://github.com/deepinsight/insightface) and [GFPGAN](https://github.com/TencentARC/GFPGAN) solutions.

## Usage

```
git clone git@github.com:ixtal23/deepdeepdopdop.git

cd deepdeepdopdop

pip install -r requirements.txt

python main.py --source-face-image-file SOURCE_FACE_IMAGE_FILE
               --input-file INPUT_FILE
               [--output-file OUTPUT_FILE] 
               [--restore-face]
               [--process-every-face]
               [--process-video-in-memory]
               [--reference-face-position REFERENCE_FACE_POSITION]
               [--reference-frame-time REFERENCE_FRAME_TIME]
               [--similar-face-distance SIMILAR_FACE_DISTANCE]
               [--execution-provider {CUDAExecutionProvider,CPUExecutionProvider}]
               [-h]
```

### Options

```
--source-face-image-file SOURCE_FACE_IMAGE_FILE                     a path to an image file with a source face a peth to an input image or video file to process
--output-file OUTPUT_FILE                                           a path to an output file
--restore-face                                                      restore face after swapping
--process-every-face                                                process every face
--process-video-in-memory                                           process video in memory
--reference-face-position REFERENCE_FACE_POSITION                   the position of the reference face
--reference-frame-time REFERENCE_FRAME_TIME                         the time of the reference frame in milliseconds
--similar-face-distance SIMILAR_FACE_DISTANCE                       a face distance used for recognition
--execution-provider {CUDAExecutionProvider,CPUExecutionProvider}   ONNX runtime execution provider
-h, --help                                                          show this help message and exit
```

## Credits

Thanks a lot all developers behind libraries used in this project:
- [InsightFace](https://github.com/deepinsight/insightface) 
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [OpenCV](https://github.com/opencv/opencv)
- [opencv-python](https://github.com/opencv/opencv-python)
- [FFMPEG](https://github.com/FFmpeg/FFmpeg)
- [PyAV](https://github.com/PyAV-Org/PyAV)
- [tqdm](https://github.com/tqdm/tqdm)
