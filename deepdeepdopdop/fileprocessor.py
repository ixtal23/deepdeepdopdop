import logging as log

from .configuration import Configuration
from .faceprocessor import FaceProcessor

class FileProcessor:
    def __init__(self, configuration : Configuration, face_processor : FaceProcessor):
        self.configuration = configuration
        self.face_processor = face_processor

    def run(self) -> None:
        pass
