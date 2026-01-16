from tensorflow.keras import layers, Input, backend, Model
from qkeras           import quantized_bits, QActivation, QDense, QDepthwiseConv2D
from modules.layers   import QSymmetricDepthwiseConv2D, SymmetricDepthwiseConv2D, EtaPhiPadding, TowerEtaPhiLayer, PushMaxWeightToUnity

backend.set_image_data_format("channels_last")

def init_qrelu(precision: int):
    return QActivation(f"quantized_relu({precision},{precision//2})")

def init_quantiser(precision: int):
    return quantized_bits(bits = precision, integer = precision // 2, alpha = 1)

def QuantisedCNN(
        size              : int,
        depth_multiplier  : int,
        hidden_layer_sizes: list[int],
        symmetric         : bool,
        conv_bits         : int,
        dense_bits        : int,
        output_layers     : list[tuple],
        input_shape       : tuple[int],
        aux_input_shape   : tuple[int] = None,
        name              : str        = "qcnn"
):
    has_aux_inputs  = aux_input_shape is not None

    conv_quantiser  = init_quantiser(conv_bits)
    dense_quantiser = init_quantiser(dense_bits)

    inputs          = Input(shape=input_shape)

    if symmetric:
        x = QSymmetricDepthwiseConv2D(
            kernel_size      = size,
            input_channels   = input_shape[-1],
            depth_multiplier = depth_multiplier,
            kernel_quantizer = conv_quantiser,
            bias_quantizer   = conv_quantiser,
        )(inputs)
    else:
        x = QDepthwiseConv2D(
            kernel_size      = (size, size),
            input_channels   = input_shape[-1],
            depth_multiplier = depth_multiplier,
            kernel_quantizer = conv_quantiser,
            bias_quantizer   = conv_quantiser,
        )(inputs)

    x = init_qrelu(conv_bits)(x)

    if has_aux_inputs:
        aux_inputs = Input(shape=aux_input_shape)
        x          = layers.Concatenate(axis=-1)([x, aux_inputs])

    for layer_size in hidden_layer_sizes:
        x = QDense(
            layer_size,
            kernel_quantizer = dense_quantiser,
            bias_quantizer   = dense_quantiser,
        )(x)
        x = init_qrelu(dense_bits)(x)

    outputs = [
        activation(
            QDense(
                output_size,
                kernel_quantizer = dense_quantiser,
                bias_quantizer   = dense_quantiser,
            )(x)
        )
        for output_size, activation in output_layers
    ]

    model = Model(
        inputs  = [inputs, aux_inputs] if has_aux_inputs else inputs,
        outputs = output[0]            if len(outputs) == 0 else outputs,
        name    = name)

    return model
