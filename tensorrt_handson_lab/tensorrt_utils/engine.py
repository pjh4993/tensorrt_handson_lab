# type: ignore
"""Tensorrt engine wrapper"""
import logging as logger
import os
import re
from typing import Dict

import numpy as np
import tensorrt as trt
from cuda import cudart

from tensorrt_handson_lab.tensorrt_utils import common


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.len = 10
        self.data_iter = None
        self.id_allocation = None
        self.mask_allocation = None
        self.cnt = 0

    def set_image_batcher(self, data_iter):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.data_iter = iter(data_iter)
        size = int(np.dtype(np.int32).itemsize * (128 * 140))
        self.id_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.mask_allocation = common.cuda_call(cudart.cudaMalloc(size))

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        return 128

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.data_iter:
            return None
        if self.cnt < self.len:
            batch = next(self.data_iter)
            common.memcpy_host_to_device(
                self.id_allocation,
                np.ascontiguousarray(batch["input_ids"].cpu().numpy().astype(np.int32)),
            )
            common.memcpy_host_to_device(
                self.mask_allocation,
                np.ascontiguousarray(batch["attention_mask"].cpu().numpy().astype(np.int32)),
            )
            self.cnt += 1
            return [int(self.id_allocation), int(self.mask_allocation)]
        else:
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        return None

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        return None


class TRTEngine:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(
        self,
        dataloader,
        verbose: bool,
        workspace: int,
        precision: str,
        engine_path: str,
        fp32_layer_pattern: list[str],
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 1,
    ):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.network = None
        self.parser = None
        self.engine = None
        self.context = None
        self.dynamic = False
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.precision = precision
        self.workspace = workspace
        self.verbose = verbose
        self.engine_path = os.path.realpath(engine_path)
        self.dataloader = dataloader
        self.batch_size = self.max_batch
        self.fp_32_layer_pattern = fp32_layer_pattern

        if (self.min_batch == self.max_batch) and (self.min_batch == self.opt_batch):
            self.dynamic = True

    def prepare_builder(self):
        """prepare tensorrt engine builder"""
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if self.verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.builder.max_batch_size = self.max_batch
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = self.workspace  # 8 GB
        self.config.avg_timing_iterations = 10
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    def create_network(self, serialized_onnx: bytes) -> bool:
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(flags=network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        if not self.parser.parse(serialized_onnx):
            logger.error("Failed to load serialized onnx")
            for error in range(self.parser.num_errors):
                logger.error(self.parser.get_error(error))
            return False

        engine_layers = [self.network.get_layer(i) for i in range(self.network.num_layers)]

        if len(self.fp_32_layer_pattern):
            self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

        def is_target_layers(x):
            return x.type in [
                trt.LayerType.CONVOLUTION,
                trt.LayerType.ELEMENTWISE,
                trt.LayerType.MATRIX_MULTIPLY,
                trt.LayerType.REDUCE,
                trt.LayerType.UNARY,
            ]

        logger.info("Network Description")
        for lidx, el in enumerate(engine_layers):
            # Strange but need at BEIT
            # if re.search(r"\/blocks\.\d+\/Add_\d+", el.name):
            #     self.network.mark_output(el.get_output(0))
            for fp32_pt in self.fp_32_layer_pattern:
                if re.search(fp32_pt, el.name) and is_target_layers(el):
                    el.precision = trt.DataType.FLOAT
                    el.set_output_type(0, trt.DataType.FLOAT)
                    break
            # if not prec_set and is_target_layers(el) and (el.precision == trt.DataType.FLOAT):
            #     el.precision = trt.DataType.HALF
            #     el.set_output_type(0, trt.DataType.HALF)

            logger.info(
                "[%04d] name : %s, type : %s, precision : %s",
                lidx,
                el.name,
                el.type,
                repr(el.precision),
            )

        engine_inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        engine_outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        opt_profile = self.builder.create_optimization_profile()

        for ei in engine_inputs:
            logger.info("Input '%s' with shape %s and dtype %s", ei.name, ei.shape, ei.dtype)
            if ei.shape[0] == -1:
                base_shape = list(ei.shape[1:])
                opt_profile.set_shape(
                    ei.name,
                    min=[self.min_batch] + base_shape,
                    opt=[self.opt_batch] + base_shape,
                    max=[self.max_batch] + base_shape,
                )
                logger.info(
                    "Input '%s' add dynamic shape %s", ei.name, repr(opt_profile.get_shape(ei.name))
                )
        for eo in engine_outputs:
            logger.info("Output '%s' with shape %s and dtype %s", eo.name, eo.shape, eo.dtype)
            # if eo.shape[0] == -1:
            #     base_shape = list(eo.shape[1:])
            #     opt_profile.set_shape(
            #         eo.name,
            #         min=[self.min_batch] + base_shape,
            #         opt=[self.opt_batch] + base_shape,
            #         max=[self.max_batch] + base_shape,
            #     )
            #     logger.info(
            #         "Output '%s' add dynamic shape %s",
            #         eo.name,
            #         repr(opt_profile.get_shape(eo.name)),
            #     )

        self.config.add_optimization_profile(opt_profile)
        return True

    def create_engine(
        self,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        """
        engine_dir = os.path.dirname(self.engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        logger.info("Building %s Engine in %s", self.precision, self.engine_path)

        if self.precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            if not self.builder.platform_has_fast_int8:
                logger.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator()
                self.config.int8_calibrator.set_image_batcher(self.dataloader.val_dataloader())

        engine = self.builder.build_engine(self.network, self.config)
        with open(self.engine_path, "wb") as f:
            logger.info("Serializing engine to file: %s", self.engine_path)
            f.write(engine.serialize())

    def _prepare_infer(self, sample: Dict[str, np.ndarray]):
        """prepare bindings for infer"""
        with open(self.engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = {}
        self.outputs = {}
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = (
                sample[name].shape
                if is_input
                else [self.batch_size] + list(self.engine.get_binding_shape(i))[1:]
            )
            self.context.set_binding_shape(i, shape)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs[name] = binding
            else:
                self.outputs[name] = binding

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_keys(self):
        return list(self.inputs.keys())

    def output_placeholder(self):
        """
        get numpy placeholder for copy result in device
        """
        outputs = {}
        for k, val in self.outputs.items():
            outputs[k] = np.zeros(dtype=val["dtype"], shape=val["shape"])
        return outputs

    def infer(self, batch: Dict[str, np.ndarray]):
        if self.engine is None:
            self._prepare_infer(batch)

        # Prepare the output data

        # Process I/O and execute the network
        batch_size = self.batch_size
        for k, val in batch.items():
            if self.inputs[k]["shape"][0] > val.shape[0]:
                padded = np.zeros(dtype=self.inputs[k]["dtype"], shape=self.inputs[k]["shape"])
                padded[: len(val)] = val
                batch_size = val.shape[0]

            common.memcpy_host_to_device(
                self.inputs[k]["allocation"],
                np.ascontiguousarray(val.astype(self.inputs[k]["dtype"])),
            )
        self.context.execute_v2(self.allocations)

        host_output = self.output_placeholder()
        for k, val in self.outputs.items():
            common.memcpy_device_to_host(host_output[k], val["allocation"])
            host_output[k] = host_output[k][:batch_size]

        return host_output
