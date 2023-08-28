import logging as log

import onnx
import onnxruntime

import insightface
import warnings
warnings.filterwarnings('ignore', category = FutureWarning, module = 'insightface')

from .configuration import Configuration
from .types import Frame, Face
from .utils import download

class FaceSwapper:
    def __init__(self, configuration : Configuration):
        self.configuration = configuration

        log.info('Prepare face swapper model')
        download(self.configuration.face_swapper_model_file_url, self.configuration.face_swapper_model_file_path)

        log.info(f'Prepare face swapper: model={self.configuration.face_swapper_model_file_path}, provider={self.configuration.execution_provider}')
        self.face_swapper = insightface.model_zoo.get_model(str(self.configuration.face_swapper_model_file_path), providers = [self.configuration.execution_provider])

        log.info(f'Set ONNX Runtime logger severity to {self.configuration.onnxruntime_logging_severity}')
        onnxruntime.set_default_logger_severity(self.configuration.onnxruntime_logging_severity)

    def process(self, source_face : Face, target_face : Face, frame : Frame) -> Frame:
        return self.face_swapper.get(frame, target_face, source_face, paste_back = True)
