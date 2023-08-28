import logging as log

from gfpgan.utils import GFPGANer
import warnings
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

from .configuration import Configuration
from .types import Frame, Face
from .utils import download

class FaceRestorer:
    def __init__(self, configuration : Configuration):
        self.configuration = configuration

        log.info('Prepare face restorer model')
        download(self.configuration.face_restorer_model_file_url, self.configuration.face_restorer_model_file_path)

        log.info(f'Prepare face restorer: model={self.configuration.face_restorer_model_file_path}, device={self.configuration.gfpgan_device}')
        self.face_restorer = GFPGANer(model_path = str(self.configuration.face_restorer_model_file_path), upscale = 1, device = self.configuration.gfpgan_device)

    def process(self, target_face : Face, frame : Frame) -> Frame:
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])

        face_for_restoration = frame[start_y : end_y, start_x : end_x]
        if face_for_restoration.size:
            _, _, restored_face = self.face_restorer.enhance(face_for_restoration, paste_back = True)
            frame[start_y : end_y, start_x : end_x] = restored_face

        return frame
