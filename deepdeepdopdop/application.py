import logging as log
import av
from contextlib import ContextDecorator
from typing import Optional

from .configuration import Configuration
from .faceprocessor import FaceProcessor
from .faceprocessor import EveryFaceProcessor
from .fileprocessor import FileProcessor
from .imageprocessor import ImageProcessor
from .videoprocessor import VideoProcessor
from .utils import is_image, is_video

class Application(ContextDecorator):
    def __init__(self):
        self.configuration = Configuration()

        log.basicConfig(level = self.configuration.log_level, format = self.configuration.log_format)

    def __enter__(self):
        log.info('Start')
        return self

    def __exit__(self, *args):
        log.info('Finish')

    def __create_face_processor(self) -> FaceProcessor:
        return EveryFaceProcessor(self.configuration) if self.configuration.process_every_face else FaceProcessor(self.configuration)

    def __create_file_processor(self, face_processor : FaceProcessor) -> Optional[FileProcessor]:
        if is_image(self.configuration.input_file):
            return ImageProcessor(self.configuration, face_processor)
        elif is_video(self.configuration.input_file):
            return VideoProcessor(self.configuration, face_processor)

        return None

    def __process(self) -> None:
        face_processor = self.__create_face_processor()
        file_processor = self.__create_file_processor(face_processor)
        if file_processor:
            file_processor.run()

    def run(self) -> None:
        if self.configuration.parse_command_line():
            self.__process()
