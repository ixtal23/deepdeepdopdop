import logging as log

import numpy

import insightface
import warnings
warnings.filterwarnings('ignore', category = FutureWarning, module = 'insightface')

from typing import Optional, List

from .configuration import Configuration
from .types import Frame, Face
from .imageio import read_image
from .videoio import VideoReader

class FaceAnalyser:
    def __init__(self, configuration : Configuration):
        self.configuration = configuration

        log.info(f'Prepare face analyser: provider={self.configuration.execution_provider}')
        self.face_analyser = insightface.app.FaceAnalysis(name = 'buffalo_l', providers = [self.configuration.execution_provider])
        self.face_analyser.prepare(ctx_id = 0, det_thresh = 0.5, det_size = (640, 640))

    def find_faces(self, frame : Frame) -> Optional[List[Face]]:
        try:
            faces = self.face_analyser.get(frame)
            return sorted(faces, key = lambda x: x.bbox[0])
        except ValueError:
            return None

    def __find_face(self, frame : Frame, position : int) -> Optional[Face]:
        faces = self.face_analyser.get(frame)
        if faces:
            try:
                return faces[position] if position >= 0 else min(faces, key = lambda x: x.bbox[0])
            except IndexError:
                return None
        return None

    def find_source_face_in_image(self) -> Optional[Face]:
        log.info(f'Find source face in image file {self.configuration.source_face_image_file}')
        source_face : Face = None

        face_image = read_image(self.configuration.source_face_image_file)
        if face_image.any():
            source_face = self.__find_face(face_image, 0)
            if source_face:
                log.info(f'Source face found: det_score={source_face["det_score"]}, gender={source_face["gender"]}, age={source_face["age"]}, bbox={source_face["bbox"]}')
            else:
                log.error('Source face not found')

        return source_face

    def find_reference_face_in_image(self, frame : Frame) -> Optional[Face]:
        log.info(f'Find reference face at position #{self.configuration.reference_face_position} in image')

        reference_face = self.__find_face(frame, self.configuration.reference_face_position)
        if reference_face:
            log.info(f'Reference face found: det_score={reference_face["det_score"]}, gender={reference_face["gender"]}, age={reference_face["age"]}, bbox={reference_face["bbox"]}')
        else:
            log.error('Reference face not found')

        return reference_face

    def find_reference_face_in_video_frame(self, frame : Frame) -> Optional[Face]:
        # No logging because this function is called in loop.
        return self.__find_face(frame, self.configuration.reference_face_position)

    def find_reference_face_in_video(self) -> Optional[Face]:
        log.info(f'Find reference face at position #{self.configuration.reference_face_position} in video frame at {self.configuration.reference_frame_time} msec')
        reference_face : Face = None

        frame = VideoReader.read_frame(self.configuration.input_file, self.configuration.reference_frame_time)
        if frame:
            imageio.write_image(f'{self.configuration.output_file}.reference_face_frame_at_{self.configuration.reference_frame_time}_msec.png', frame)

            reference_face = self.find_reference_face_in_video_frame(frame)
            if reference_face:
                log.info(f'Reference face found in frame at {self.configuration.reference_frame_time} msec: det_score={reference_face["det_score"]}, gender={reference_face["gender"]}, age={reference_face["age"]}, bbox={reference_face["bbox"]}')
            else:
                log.error(f'Reference face not found in frame at {self.configuration.reference_frame_time} msec')

        return reference_face

    def find_similar_face(self, frame : Frame, reference_face : Face) -> Optional[Face]:
        faces = self.find_faces(frame)
        if faces:
            for face in faces:
                if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                    distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                    if distance < self.configuration.similar_face_distance:
                        return face
        return None
