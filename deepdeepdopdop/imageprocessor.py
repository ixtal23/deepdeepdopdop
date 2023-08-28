import logging as log

from .configuration import Configuration
from .fileprocessor import FileProcessor
from .faceprocessor import FaceProcessor
from .imageio import read_image, write_image

class ImageProcessor(FileProcessor):
    def __init__(self, configuration : Configuration, face_processor : FaceProcessor):
        super().__init__(configuration, face_processor)

    def run(self) -> None:
        log.info(f'Process input image file {self.configuration.input_file}')

        source_face = self.face_processor.face_analyser.find_source_face_in_image()
        if source_face:
            input_image = read_image(self.configuration.input_file)
            if input_image.any():
                reference_face = self.face_processor.face_analyser.find_reference_face_in_image(input_image)
                if reference_face:
                    log.info('Processing faces...')
                    output_image = self.face_processor.process(source_face, reference_face, input_image)

                    log.info('Write result into output file')
                    write_image(self.configuration.output_file, output_image)
