"""Export a checkpoint as an ONNX model.

Applies onnx utilities to improve the exported model and
also tries to simplify the model with onnx-simplifier.

https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
https://github.com/daquexian/onnx-simplifier
"""

import argparse
import logging
import shutil

import torch

import openpifpaf

try:
    import onnx
    import onnx.utils
except ImportError:
    onnx = None

try:
    import onnxsim
except ImportError:
    onnxsim = None

LOG = logging.getLogger(__name__)


def image_size_warning(basenet_stride, input_w, input_h):
    if input_w % basenet_stride != 1:
        LOG.warning(
            'input width (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_w, basenet_stride,
            (input_w - 1) // basenet_stride * basenet_stride + 1,
            ((input_w - 1) // basenet_stride + 1) * basenet_stride + 1,
        )

    if input_h % basenet_stride != 1:
        LOG.warning(
            'input height (%d) should be a multiple of basenet '
            'stride (%d) + 1: closest are %d and %d',
            input_h, basenet_stride,
            (input_h - 1) // basenet_stride * basenet_stride + 1,
            ((input_h - 1) // basenet_stride + 1) * basenet_stride + 1,
        )


def apply(model,
          outfile,
          verbose=True,
          input_w=129,
          input_h=97,
          static=False,
          opset_version=11):
    image_size_warning(model.base_net.stride, input_w, input_h)

    # configure
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    input_names=['input_batch']
    output_names=['cif', 'caf']
    dummy_input = torch.randn(1, 3, input_h, input_w)

    if static:
        torch.onnx.export(
            model,
            dummy_input,
            outfile,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            # keep_initializers_as_inputs=True,
            opset_version=opset_version,
            do_constant_folding=True)
    else:
        dynamic_axes={}
        dynamic_label='dynamic'
        for tensor in input_names+output_names:
            dynamic_axes[tensor] = {0: dynamic_label}

        torch.onnx.export(
            model,
            dummy_input,
            outfile,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            # keep_initializers_as_inputs=True,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes)

    if onnx is not None:
        # Keep only the actual input(s) and outputs ( = remove caf25 branch)
        onnx.utils.extract_model(outfile, outfile, input_names, output_names)
        if not static:
            from onnx.tools import update_model_dims
            onnx_model_dyn = onnx.load(outfile)
            #https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#updating-models-inputs-outputs-dimension-sizes-with-variable-length
            variable_length_model = update_model_dims.update_inputs_outputs_dims(onnx_model_dyn, 
                                                                                {input_names[0]: [dynamic_label]}, 
                                                                                {output_names[0]: [dynamic_label], output_names[1]: [dynamic_label]}
                                                                                )
            onnx.save_model(variable_length_model, outfile)

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    add_const_value_infos_to_graph(model.graph)
    return model


def optimize(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unoptimized.onnx')
        shutil.copyfile(outfile, infile)

    all_passes = onnx.optimizer.get_available_passes()
    print("Available optimization passes:")
    for p in all_passes:
        print('\t{}'.format(p))
    print()
    #passes = ['fuse_consecutive_transposes', 'fuse_bn_into_conv', 'fuse_add_bias_into_conv']
    passes = [  'fuse_add_bias_into_conv',
                'fuse_bn_into_conv']

    model = onnx.load(infile)
    onnx.checker.check_model(model)
    if False:
        inferred_model = onnx.shape_inference.infer_shapes(model)
        inferred_model = add_value_info_for_constants(inferred_model)
        for init in model.graph.initializer:
            for value_info in model.graph.value_info:
                if init.name == value_info.name:
                    inferred_model.graph.input.append(value_info)
        print(inferred_model.graph.value_info)
        print(onnx.helper.printable_graph(inferred_model.graph))
        optimized_model = onnx.optimizer.optimize(inferred_model, passes=['fuse_bn_into_conv'])
    else:
        # https://github.com/onnx/onnx/issues/3219#issuecomment-761618519
        optimized_model = onnx.optimizer.optimize(model, passes)
    #optimized_model = onnx.optimizer.optimize(model, passes)
    print('The model after optimization:\n\n{}'.format(onnx.helper.printable_graph(optimized_model.graph)))
    onnx.checker.check_model(optimized_model)

    onnx.save(optimized_model, outfile)


def check(modelfile):
    model = onnx.load(modelfile)
    onnx.checker.check_model(model)


def polish(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unpolished.onnx')
        shutil.copyfile(outfile, infile)

    model = onnx.load(infile)
    polished_model = onnx.utils.polish_model(model)
    onnx.save(polished_model, outfile)


def simplify(infile, outfile=None):
    if outfile is None:
        assert infile.endswith('.onnx')
        outfile = infile
        infile = infile.replace('.onnx', '.unsimplified.onnx')
        shutil.copyfile(outfile, infile)

    simplified_model, check_ok = onnxsim.simplify(infile, check_n=3, perform_optimization=False)
    assert check_ok
    onnx.save(simplified_model, outfile)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.export_onnx',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    openpifpaf.network.Factory.cli(parser)

    parser.add_argument('--outfile', default='openpifpaf-resnet50.onnx')
    parser.add_argument('--simplify', dest='simplify', default=False, action='store_true')
    parser.add_argument('--polish', dest='polish', default=False, action='store_true',
                        help='runs checker, optimizer and shape inference')
    parser.add_argument('--optimize', dest='optimize', default=False, action='store_true')
    parser.add_argument('--static', dest='static', default=False, action='store_true')
    parser.add_argument('--check', dest='check', default=False, action='store_true')
    parser.add_argument('--input-width', type=int, default=129)
    parser.add_argument('--input-height', type=int, default=97)
    args = parser.parse_args()

    with torch.no_grad():
        openpifpaf.network.Factory.configure(args)

        model, _ = openpifpaf.network.Factory().factory()

        apply(model, args.outfile, input_w=args.input_width, input_h=args.input_height, static=args.static)
        if args.simplify:
            simplify(args.outfile)
        if args.optimize:
            optimize(args.outfile)
        if args.polish:
            polish(args.outfile)
        if args.check:
            check(args.outfile)


if __name__ == '__main__':
    main()
