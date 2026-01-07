from tensorflow.keras import layers, initializers, backend, regularizers, Model, Input
from qkeras           import QDense, quantized_bits

import keras
import tensorflow as tf
import numpy      as np

class SymmetricPooling(layers.Layer):
    def __init__(self, size: int, input_channels: int):
        super().__init__()
        if size % 2 != 1:
            raise ValueError("size must be odd integer")

        centre      = size // 2
        n_features  = (centre + 1) ** 2  #
        k           = np.zeros((size, size, 1, n_features), dtype=np.float32)
        feature_idx = 0
        for eta_idx in range(centre + 1):
            for phi_idx in range(centre + 1):
                for i, j in itertools.product(
                    [eta_idx, size - 1 - eta_idx], [phi_idx, size - 1 - phi_idx]
                ):
                    k[i, j, :, feature_idx] = 1
                feature_idx += 1

        assert feature_idx == n_features

        self.kernel = tf.constant(np.repeat(k, input_channels, axis=2))

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(
            inputs, self.kernel, strides=[1] * 4, padding="VALID"
        )

class SymmetricDepthwiseConv2D(layers.Layer):
    def __init__(
            self,
            kernel_size     : int,
            depth_multiplier: int,
            input_channels  : int = 6,
            **kwargs
    ):
        super().__init__()

        self.kernel_size      = kernel_size
        self.input_channels   = input_channels
        self.depth_multiplier = depth_multiplier

        self.pooling      = SymmetricPooling(size=kernel_size, input_channels=input_channels)
        self.dense_layers = []
        for _ in range(input_channels):
            self.dense_layers.append(layers.Dense(depth_multiplier))

    def call(self, inputs):
        pooled_inputs          = self.pooling(inputs)
        pooled_inputs_by_layer = tf.split(pooled_inputs, self.input_channels, axis=-1)
        pooled_inputs_by_layer = [
            dense_layer(x)
            for dense_layer, x in zip(self.dense_layers, pooled_inputs_by_layer)
        ]
        outputs = layers.Concatenate()(pooled_inputs_by_layer)
        return outputs

    def get_config(self):
        base_config = super().get_config()
        config      = {
            "kernel_size"     : self.kernel_size,
            "depth_multiplier": self.depth_multiplier,
            "dense_layers"    : keras.saving.serialize_keras_object(self.dense_layers),
            "input_channels"  : self.input_channels,
        }
        return {**base_config, **config}


class QSymmetricDepthwiseConv2D(SymmetricDepthwiseConv2D):
    def __init__(
        self,
        kernel_size     : int,
        kernel_quantizer: quantized_bits,
        bias_quantizer  : quantized_bits,
        depth_multiplier: int = 1,
        input_channels  : int = 6,
        **kwargs,
    ):
        super().__init__(
            kernel_size      = kernel_size,
            depth_multiplier = depth_multiplier,
            input_channels   = input_channels,
            **kwargs,
        )

        self.bias_quantizer   = bias_quantizer
        self.kernel_quantizer = kernel_quantizer
        self.dense_layers     = []
        for _ in range(input_channels):
            self.dense_layers.append(
                QDense(
                    depth_multiplier,
                    kernel_quantizer = kernel_quantizer,
                    bias_quantizer   = bias_quantizer,
                )
            )

    def get_config(self):
        base_config = super().get_config()
        config      = {
            "bias_quantizer"  : keras.saving.serialize_keras_object(self.bias_quantizer),
            "kernel_quantizer": keras.saving.serialize_keras_object(
                self.kernel_quantizer
            ),
        }
        return {**base_config, **config}

class PushMaxWeightToUnity(regularizers.Regularizer):
    def __init__(self,
                 strength: float,
                 axis    : int = (1, 2, 3),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.strength = strength
        self.axis     = axis

    def __call__(self, w):
        penalty = backend.abs(backend.max(w, axis=self.axis) - 1.0)
        return self.strength * backend.mean(penalty)

    def get_config(self):
        return {"strength": self.strength, "axis": self.axis}

    
class TowerEtaPhiLayer(layers.Layer):
    def __init__(self, deta: float = 0.1, dphi: float = np.pi / 32, **kwargs):
        super().__init__(**kwargs)
        self.deta = deta
        self.dphi = dphi

    def call(self, image):
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta_idxs = tf.tile(tf.reshape(tf.range(E), (1, E, 1, 1)), (B, 1, P, 1))
        eta      = tf.cast(2 * eta_idxs - E + 1, dtype=tf.float32) * self.deta / 2.0

        phi_idxs = tf.tile(tf.reshape(tf.range(P), (1, 1, P, 1)), (B, E, 1, 1))
        phi      = tf.cast(2 * phi_idxs - P + 1, dtype=tf.float32) * self.dphi / 2.0

        return eta, phi

    def get_config(self):
        return {**super().get_config(), "deta": self.deta, "dphi": self.dphi}

class ImageToMomentumList(layers.Layer):
    def __init__(self, max_vectors: int = 20, min_pt: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_vectors = max_vectors
        self.min_pt      = min_pt
        self.get_coords  = TowerEtaPhiLayer()

    def call(self, image):
        B, E, P, _ = tf.unstack(tf.shape(image))

        eta, phi = self.get_coords(image)
        pt       = backend.sum(image, axis=-1, keepdims=True)

        flat_pt  = tf.reshape(pt, [B, E * P])
        flat_eta = tf.reshape(eta, [B, E * P])
        flat_phi = tf.reshape(phi, [B, E * P])

        seed_pt, idxs = tf.math.top_k(flat_pt, k=self.max_vectors, sorted=True)

        seed_eta = tf.gather(flat_eta, idxs, axis=1, batch_dims=1)
        seed_phi = tf.gather(flat_phi, idxs, axis=1, batch_dims=1)

        seed_mask = seed_pt > self.min_pt
        seed_pt   = tf.where(seed_mask, seed_pt, 0)
        seed_eta  = tf.where(seed_mask, seed_eta, 0)
        seed_phi  = tf.where(seed_mask, seed_phi, 0)

        return tf.stack([seed_pt, seed_eta, seed_phi], axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "max_vectors": self.max_vectors,
            "min_pt": self.min_pt,
        }

class EtaPhiPadding(layers.Layer):
    def __init__(self, pad_size, **kwargs):
        super().__init__(**kwargs)
        self.pad_size = pad_size

    def cyclic_padding_at_axis(self, x, axis=2):
        length     = tf.shape(x)[axis]
        pad_before = tf.gather(x, tf.range(length - self.pad_size, length), axis=axis)
        pad_after  = tf.gather(x, tf.range(0, self.pad_size), axis=axis)
        return tf.concat([pad_before, x, pad_after], axis=axis)

    def zero_padding_at_axis(self, x, axis=1):
        rank           = len(x.shape)
        paddings       = [[0, 0]] * rank
        paddings[axis] = [self.pad_size, self.pad_size]
        return tf.pad(x, paddings, mode="CONSTANT", constant_values=0)

    def call(self, x):
        return self.zero_padding_at_axis(self.cyclic_padding_at_axis(x))

    def get_config(self):
        return {**super().get_config(), "pad_size": self.pad_size}

    
class SlidingConeSum(layers.Layer):
    def __init__(
        self,
        kernel_size: int = 7,
        shape      : str = "circle",
        radius     : int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = int(kernel_size)
        self.shape       = shape
        self.radius      = radius

        self.kernel = self.init_kernel()
        self.pad    = EtaPhiPadding(pad_size=kernel_size // 2)

    def call(self, x):
        return tf.nn.conv2d(self.pad(x), self.kernel, strides=1, padding="VALID")

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_size": self.kernel_size,
            "shape"      : self.shape,
            "radius"     : self.radius,
        }

    def init_kernel(self) -> np.array:
        ks = int(self.kernel_size)
        if self.shape == "square":
            kernel = np.ones((ks, ks, 1, 1), dtype=np.float32)
            return tf.convert_to_tensor(kernel)

        elif self.shape == "circle":
            c      = ks // 2
            r      = c if self.radius is None else self.radius
            yy, xx = np.ogrid[:ks, :ks]
            mask   = ((yy - c) ** 2 + (xx - c) ** 2) <= (r**2)
            kernel = mask.astype(np.float32)[..., None, None]
            return tf.convert_to_tensor(kernel)
        else:
            raise ValueError("Shape must be 'square' or 'circle'")


class LocalMaxMask(layers.Layer):
    def __init__(
        self,
        kernel_size  : int = 7,
        tie_eps_input: float = 1e-3,
        tie_eps_pos  : float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pad        = EtaPhiPadding(pad_size=kernel_size // 2)
        self.pool       = layers.MaxPool2D(
            pool_size=(kernel_size, kernel_size), strides=(1, 1), padding="VALID"
        )
        self.tie_eps_input = tie_eps_input
        self.tie_eps_pos   = tie_eps_pos

    def call(self, image):
        if isinstance(image, (tuple, list)) and len(image) == 2:
            sum_map, raw = image
        else:
            sum_map = image
            raw     = None

        dtype = sum_map.dtype
        shp   = tf.shape(sum_map)
        h     = shp[1]
        w     = shp[2]

        h_f      = tf.cast(h, dtype)
        w_f      = tf.cast(w, dtype)
        r        = tf.cast(tf.range(h)[:, None], dtype)
        c        = tf.cast(tf.range(w)[None, :], dtype)
        pos_ramp = (r * (w_f + 1) + c) / tf.cast(h_f * (w_f + 1), dtype)
        pos_ramp = tf.reshape(pos_ramp, [1, h, w, 1])

        if raw is not None:
            raw_map = tf.reduce_sum(raw, axis=-1, keepdims=True)
            score   = sum_map * raw_map
        else:
            score = sum_map

        scale = tf.maximum(
            tf.reduce_max(tf.abs(score), axis=[1, 2, 3], keepdims=True),
            tf.constant(1.0, dtype=dtype),
        )

        score  = score + self.tie_eps_pos * scale * pos_ramp

        pooled = self.pool(self.pad(score))
        is_max = tf.equal(score, pooled)

        return tf.where(is_max, sum_map, tf.zeros_like(sum_map))

    def get_config(self):
        return {
            **super().get_config(),
            "kernel_size"  : self.kernel_size,
            "tie_eps_input": self.tie_eps_input,
            "tie_eps_pos"  : self.tie_eps_pos,
        }
    
