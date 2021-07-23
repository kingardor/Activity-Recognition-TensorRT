import argparse

def parse_arguments():
        """ Method to parse arguments using argparser. """

        parser = argparse.ArgumentParser(description='Action Recognition using TensorRT 8')
        parser.add_argument('--stream', type=str, default='', help='Path to use video stream')
        parser.add_argument('--model', type=str, default='', help='Path to model to use')
        parser.add_argument('--fp16', default=False, action='store_true', help='To enable fp16 precision')
        parser.add_argument('--frameskip', type=int, default=2, help='Number of frames to skip')
        parser.add_argument('--save_output', type=str, default='', help='Save output as video')

        return parser.parse_args()