import argparse
import os
import json
import time
import logging
import torch
from pathlib import Path

from effdet import create_model

# turn off JIT optimizations and make layers exportable
import timm.models.layers.config
timm.models.layers.config.set_exportable(True)
timm.models.layers.config.set_no_jit(True)

# turn of JIT optimizations in TIMM module
#  -> otherwise the activation functions cannot be exported to ONNX

parser = argparse.ArgumentParser(description='PyTorch ImageNet Exporter')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-o', '--out_file', default='tmp.onnx',
                    help='output file path')

def valid_tensor(s):
    msg = "Not a valid resolution: '{0}' [CxHxW].".format(s)
    try:
        q = s.split('x')
        if len(q) != 3:
            raise argparse.ArgumentTypeError(msg)
        return [int(v) for v in q]
    except ValueError:
        raise argparse.ArgumentTypeError(msg)
parser.add_argument('-r', '--ONNX_resolution', default="3x512x512", type=valid_tensor,
                    help='ONNX input resolution (default: 3x223x223 [imagenet])')

def export(args):

    # creat output dir
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    # create model
    bench = create_model(
        args.model,
        bench_task='',
        checkpoint_path=args.checkpoint
    )

    bench.eval()

    # make dummy run (really required??)
    dummy_input = torch.randn([1]+[3, 512, 512])
    bench(dummy_input)

    # Export ONNX file
    input_names = [ "input:0" ]  # this are our standardized in/out nameing (required for runtime)
    output_names = [ "output:0", "output:1" ]

    print("Exporting ONNX with input resolution of {} to '{}'".format(args.ONNX_resolution,args.out_file))
    torch.onnx._export(bench, dummy_input, args.out_file,  opset_version=11, keep_initializers_as_inputs=True, output_names=output_names)
    #torch.onnx._export(bench, dummy_input, args.out_file, keep_initializers_as_inputs=True, output_names=output_names)

    print("Saved to {}".format(args.out_file))

def main():
    args = parser.parse_args()
    export(args)

if __name__ == '__main__':
    main()
