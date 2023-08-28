import logging as log

import os
import argparse
import onnxruntime

from pathlib import Path

from .utils import is_image, is_video

class Configuration:
    def __init__(self):
        self.source_face_image_file : Path = None
        self.input_file : Path = None
        self.output_file : Path = None

        self.restore_face : bool = False
        self.process_every_face : bool = False
        self.process_video_in_memory : bool = False

        self.reference_face_position : int = 0
        self.reference_frame_time : int = 0

        self.similar_face_distance : float = 0.85

        self.execution_provider : str = None
        self.gfpgan_device : str = None

        self.face_swapper_model_file_url : str = 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx'
        self.face_swapper_model_file_path : Path = Path('./model/inswapper_128.onnx')

        self.face_restorer_model_file_url : str = 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.pth'
        self.face_restorer_model_file_path : Path = Path('./model/GFPGANv1.4.pth')

        # https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
        # enum class Severity {
        #     kVERBOSE = 0,
        #     kINFO = 1,
        #     kWARNING = 2,
        #     kERROR = 3,
        #     kFATAL = 4
        # };
        self.onnxruntime_logging_severity : int = 2

        self.log_level = log.DEBUG
        self.log_format : str = '%(asctime)s   %(levelname)s   %(message)s'

    def parse_command_line(self) -> bool:
        log.info('Parse command line')

        parser = argparse.ArgumentParser(
            prog = 'deepdeepdobdob',
            description = 'Deepfake tool.',
            formatter_class = lambda prog : argparse.HelpFormatter(prog, max_help_position = 100)
        )

        parser.add_argument('--source-face-image-file', help = 'a path to an image file with a source face', dest = 'source_face_image_file', type = Path, required = True)
        parser.add_argument('--input-file', help = 'a peth to an input image or video file to process', dest = 'input_file', type = Path, required = True)
        parser.add_argument('--output-file', help = 'a path to an output file', dest = 'output_file', type = Path)

        parser.add_argument('--restore-face', help = 'restore face after swapping', dest = 'restore_face', action = 'store_true')
        parser.add_argument('--process-every-face', help = 'process every face', dest = 'process_every_face', action = 'store_true')
        parser.add_argument('--process-video-in-memory', help = 'process video in memory', dest = 'process_video_in_memory', action = 'store_true')

        parser.add_argument('--reference-face-position', help = 'the position of the reference face', dest = 'reference_face_position', type = int, default = 0)
        parser.add_argument('--reference-frame-time', help = 'the time of the reference frame in milliseconds', dest = 'reference_frame_time', type = int, default = -1)
        parser.add_argument('--similar-face-distance', help = 'a face distance used for recognition', dest = 'similar_face_distance', type = float, default = 0.85)
        
        execution_providers = onnxruntime.get_available_providers();
        default_execution_provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in execution_providers else 'CPUExecutionProvider'

        parser.add_argument('--execution-provider', help = 'ONNX runtime execution provider', dest = 'execution_provider', default = default_execution_provider, choices = execution_providers)

        args = parser.parse_args()

        self.source_face_image_file = args.source_face_image_file
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.restore_face = args.restore_face
        self.process_every_face = args.process_every_face
        self.process_video_in_memory = args.process_video_in_memory
        self.reference_face_position = args.reference_face_position
        self.reference_frame_time = args.reference_frame_time
        self.similar_face_distance = args.similar_face_distance
        self.execution_provider = args.execution_provider

        if not self.output_file:
            postfix = 'swapped-restored' if self.restore_face else 'swapped'
            self.output_file = self.input_file.with_stem(f'{video_input_file_path.stem}-{postfix}')

        self.gfpgan_device = 'cpu'
        if 'CUDAExecutionProvider' == self.execution_provider:
            self.gfpgan_device = 'cuda'
        elif 'CoreMLExecutionProvider' == self.execution_provider:
            self.gfpgan_device = 'mps'

        return self.__validate();

    def __validate(self) -> bool:
        log.info('Validate configuration')

        if not self.source_face_image_file.exists():
            log.error(f'Source face image file {self.source_face_image_file} does not exist')
            return False

        if not is_image(self.source_face_image_file):
            log.error(f'Source face image file {self.source_face_image_file} is not image')
            return False

        if not self.input_file.exists():
            log.error(f'Input file {self.input_file} does not exist')
            return False

        if not is_image(self.input_file) and not is_video(self.input_file):
            log.error(f'Input file {self.input_file} is not image or video')
            return False

        if self.output_file.exists():
            log.error(f'Output file {self.output_file} already exists')
            return False

        log.info('Configuration is ok')
        return True
