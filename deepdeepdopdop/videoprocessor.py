import logging as log

import numpy

from tqdm import tqdm

from .configuration import Configuration
from .fileprocessor import FileProcessor
from .faceprocessor import FaceProcessor
from .types import Frame, Frames, Face, TargetFaces
from .videoio import VideoReader
from .videoio import VideoWriter
from .videoio import AudioVideoMixer

class VideoProcessor(FileProcessor):
    def __init__(self, configuration : Configuration, face_processor : FaceProcessor):
        super().__init__(configuration, face_processor)

    def __process(self, video_reader : VideoReader, video_writer : VideoWriter, source_face : Face, reference_face : Face) -> None:
        with tqdm(desc = 'Processing frames', total = video_reader.frame_count, unit = 'frames') as progress:
            for input_frame in video_reader:
                if self.configuration.reference_frame_time < 0:
                    reference_face = self.face_processor.face_analyser.find_reference_face_in_video_frame(input_frame)

                if reference_face:
                    output_frame = self.face_processor.process(source_face, reference_face, input_frame)
                    video_writer.write(output_frame)
                else:
                    video_writer.write(input_frame)

                progress.update(1)

    def __process_in_memory(self, video_reader : VideoReader, video_writer : VideoWriter, source_face : Face, reference_face : Face) -> None:
        frames = video_reader.read_all()

        target_faces = self.face_processor.analyze(frames, reference_face)

        self.face_processor.swap(frames, target_faces, source_face)

        if self.configuration.restore_face:
            self.face_processor.restore(frames, target_faces)

        video_writer.write_all(frames)

    def run(self) -> None:
        log.info(f'Process input video file {self.configuration.input_file}')

        source_face = self.face_processor.face_analyser.find_source_face_in_image()
        if not source_face:
            return

        reference_face : Face = None
        if self.configuration.reference_frame_time >= 0:
            reference_face = elf.face_processor.face_analyser.find_reference_face_in_video()
            if not reference_face:
                return

        restore_audio : bool = False

        with VideoReader(self.configuration.input_file) as video_reader:
            if video_reader:
                with VideoWriter(self.configuration.output_file, video_reader.fourcc, video_reader.fps, video_reader.frame_width, video_reader.frame_height) as video_writer:
                    if video_writer:
                        if self.configuration.process_video_in_memory:
                            self.__process_in_memory(video_reader, video_writer, source_face, reference_face)
                        else:
                            self.__process(video_reader, video_writer, source_face, reference_face)
                        restore_audio = True

        if restore_audio:
            with AudioVideoMixer(self.configuration.input_file, self.configuration.output_file) as audio_video_mixer:
                if audio_video_mixer:
                    audio_video_mixer.mix()
