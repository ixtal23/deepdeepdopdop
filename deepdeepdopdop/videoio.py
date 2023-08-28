import logging as log

import cv2
import av

from contextlib import ContextDecorator
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .types import Frame, Frames

class VideoReader(ContextDecorator):
    def __init__(self, file_path : Path):
        self.file_path = file_path
        self.video_capture : cv2.VideoCapture = None
        self.fourcc : int = 0
        self.fps : float = 0
        self.frame_count : int = 0
        self.frame_width : int = 0
        self.frame_height : int = 0
        self.frame : Frame = None

    def __enter__(self):
        log.info(f'Open video file {self.file_path} for reading')

        video_capture = cv2.VideoCapture(str(self.file_path), cv2.CAP_ANY)
        if video_capture.isOpened():
            self.video_capture = video_capture

            self.fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
            self.fps = float(video_capture.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
            log.info(f'Video file was opened by OpenCV backend {video_capture.getBackendName()}: fourcc={self.fourcc}, fps={self.fps}, frame_count={self.frame_count}, frame_width={self.frame_width}, frame_height={self.frame_height}')
        else:
            log.error('Failed to open video file')

        return self

    def __exit__(self, *args):
        if self.video_capture != None:
            self.video_capture.release()

    def __bool_(self) -> bool:
        return self.video_capture != None and self.video_capture.isOpened()

    def __iter__(self):
        return self

    def __next__(self):
        if self.read():
            return self.frame
        else:
            raise StopIteration

    def get_position(self) -> int:
        return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))

    def set_position(self, time : int) -> None:
        if not self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time):
            log.error(f'CAP_PROP_POS_MSEC property is supported by OpenCV backend {self.video_capture.getBackendName()}')

    def read(self) -> bool:
        result, self.frame = self.video_capture.read()
        return result

    def read_at(self, time : int) -> bool:
        if self.set_position(time):
            return self.read()
        return false

    def read_all(self) -> Frames:
        frames : Frames = []

        with tqdm(desc = 'Read video frames', total = self.frame_count, unit = 'frames') as progress:
            for frame in self:
                frames.append(frame)
                progress.update(1)

        return frames

    @staticmethod
    def read_frame(file_path : Path, time : int) -> Optional[Frame]:
        log.info(f'Read frame from video file {file_path} at {time} msec')

        with VideoReader(file_path) as video_reader:
            if video_reader:
                if video_reader.read_at(time):
                    return video_reader.frame
                else:
                    log.error(f'Video has no frame at {time} msec')

        return None

class VideoWriter(ContextDecorator):
    def __init__(self, file_path : Path, fourcc : int, fps : float, frame_width : int, frame_height : int):
        self.file_path = file_path
        self.video_writer : cv2.VideoWriter = None
        self.fourcc : int = fourcc
        self.fps : float = fps
        self.frame_width : int = frame_width
        self.frame_height : int = frame_height

    def __enter__(self):
        log.info(f'Open video file {self.file_path} for writing: fourcc={self.fourcc}, fps={self.fps}, frame_width={self.frame_width}, frame_height={self.frame_height}')

        video_writer = cv2.VideoWriter(str(self.file_path), self.fourcc, self.fps, (self.frame_width, self.frame_height))
        if video_writer.isOpened():
            self.video_writer = video_writer
            log.info(f'Video file was opened by OpenCV backend {video_writer.getBackendName()}')
        else:
            log.error('Failed to open video file')

        return self

    def __exit__(self, *args):
        if self.video_writer != None:
            self.video_writer.release()

    def __bool_(self) -> bool:
        return self.video_writer != None and self.video_writer.isOpened()

    def write(self, frame : Frame) -> None:
        self.video_writer.write(frame)

    def write_all(self, frames: Frames) -> None:
        with tqdm(desc = 'Write video frames', total = len(frames), unit = 'frames') as progress:
            for frame in frames:
                self.write(frame)
                progress.update(1)

class AudioVideoMixer(ContextDecorator):
    def __init__(self, audio_input_file_path : Path, video_input_file_path : Path):
        self.audio_input_file_path = audio_input_file_path
        self.video_input_file_path = video_input_file_path
        self.output_file_path = video_input_file_path.with_stem(f'{video_input_file_path.stem}-with-audio')
        self.audio_input_container : av.container.InputContainer = None
        self.video_input_container : av.container.InputContainer = None
        self.output_container : av.container.OutputContainer = None

    def __enter__(self):
        log.info(f'Open video files {self.audio_input_file_path}, {self.video_input_file_path} for merging to {self.output_file_path}')
        self.audio_input_container = av.open(str(self.audio_input_file_path), mode = 'r')
        self.video_input_container = av.open(str(self.video_input_file_path), mode = 'r')
        self.output_container = av.open(str(self.output_file_path), mode = 'w')
        return self

    def __exit__(self, *args):
        if self.audio_input_container:
            self.audio_input_container.close()
        if self.video_input_container:
            self.video_input_container.close()
        if self.output_container:
            self.output_container.close()

    def __bool_(self) -> bool:
        return self.audio_input_container != None and self.video_input_container != None and self.output_container != None

    def mix(self) -> None:
        log.info(f'Mix audio stream from {self.audio_input_file_path} and video stream from {self.video_input_file_path} to {self.output_file_path}')

        input_audio_stream = self.audio_input_container.streams.audio[0]
        input_video_stream = self.video_input_container.streams.video[0]

        output_audio_stream = self.output_container.add_stream(template = input_audio_stream)
        output_video_stream = self.output_container.add_stream(template = input_video_stream)

        # Remux audio stream.
        for packet in self.audio_input_container.demux(input_audio_stream):
            # We need to skip the "flushing" packets that `demux` generates.
            if packet.dts is None:
                continue

            # We need to assign the packet to the new stream.
            packet.stream = output_audio_stream

            self.output_container.mux(packet)

        # Remux video stream.
        for packet in self.video_input_container.demux(input_video_stream):
            # We need to skip the "flushing" packets that `demux` generates.
            if packet.dts is None:
                continue

            # We need to assign the packet to the new stream.
            packet.stream = output_video_stream

            self.output_container.mux(packet)
