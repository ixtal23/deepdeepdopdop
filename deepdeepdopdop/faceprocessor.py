import logging as log

from tqdm import tqdm

from .configuration import Configuration
from .types import Frame, Frames, Face, TargetFaces
from .faceanalyser import FaceAnalyser
from .faceswapper import FaceSwapper
from .facerestorer import FaceRestorer

class FaceProcessor:
    def __init__(self, configuration : Configuration):
        self.configuration = configuration

        log.info('Prepare face processors')

        self.face_analyser = FaceAnalyser(configuration)
        self.face_swapper = FaceSwapper(configuration)
        self.face_restorer = FaceRestorer(configuration)

    def process_frame(self, source_face : Face, target_face : Face, frame : Frame) -> Frame:
        frame = self.face_swapper.process(source_face, target_face, frame)
        frame = self.face_restorer.process(target_face, frame)
        return frame

    def process(self, source_face : Face, reference_face : Face, frame : Frame) -> Frame:
        target_face = self.face_analyser.find_similar_face(frame, reference_face)
        if target_face:
            return self.process_frame(source_face, target_face, frame)
        return frame

    def analyze(self, frames : Frames, reference_face : Face) -> None:
        target_faces : TargetFaces = []
        frame_index : int = 0

        with tqdm(desc = 'Analyzing faces', total = len(frames), unit = 'frames') as progress:
            for frame in frames:
                if self.configuration.reference_frame_time < 0:
                    reference_face = self.face_analyser.find_reference_face_in_video_frame(frame)

                if reference_face:
                    target_face = self.face_analyser.find_similar_face(frame, reference_face)
                    if target_face:
                        target_faces.append((frame_index, [target_face]))

                frame_index += 1

                progress.update(1)

        return target_faces

    def swap(self, frames : Frames, target_faces : TargetFaces, source_face : Face) -> None:
        with tqdm(desc = 'Swaping faces', total = len(target_faces), unit = 'frames') as progress:
            for target_face_tuple in target_faces:
                frame_index = target_face_tuple[0]
                frame_target_faces = target_face_tuple[1]
            
                for target_face in frame_target_faces:
                    frames[frame_index] = self.face_swapper.process(source_face, target_face, frames[frame_index])

                frame_index += 1

                progress.update(1)

    def restore(self, frames : Frames,  target_faces : TargetFaces) -> None:
        with tqdm(desc = 'Restoring faces', total = len(target_faces), unit = 'frames') as progress:
            for target_face_tuple in target_faces:
                frame_index = target_face_tuple[0]
                frame_target_faces = target_face_tuple[1]

                for target_face in frame_target_faces:
                    frames[frame_index] = self.face_restorer.process(target_face, frames[frame_index])

                frame_index += 1

                progress.update(1)

class EveryFaceProcessor(FaceProcessor):
    def __init__(self, configuration : Configuration):
        super().__init__(configuration)

    def process(self, source_face : Face, reference_face : Face, frame : Frame) -> Frame:
        target_faces = self.face_analyser.find_faces(frame)
        if target_faces:
            for target_face in target_faces:
                frame = self.process_frame(source_face, target_face, frame)
        return frame

    def analyze(self, frames : Frames, reference_face : Face) -> None:
        target_faces : TargetFaces = []
        frame_index : int = 0

        with tqdm(desc = 'Analyzing faces', total = len(frames), unit = 'frames') as progress:
            for frame in frames:
                if self.configuration.reference_frame_time < 0:
                    reference_face = self.face_analyser.find_reference_face_in_video_frame(frame)

                if reference_face:
                    frame_target_faces = self.face_analyser.find_faces(frame)
                    if frame_target_faces:
                        target_faces.append((frame_index, frame_target_faces))

                frame_index += 1

                progress.update(1)

        return target_faces
