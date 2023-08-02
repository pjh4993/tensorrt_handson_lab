# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import json
import sys
import asyncio
import concurrent.futures

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import traceback
from turbojpeg import TurboJPEG, TJPF_RGB

import asyncio
from functools import partial, wraps
from contextlib import ContextDecorator

class ExceptionLogger(ContextDecorator):
    """Context Decorator for checking timestamp difference of block"""

    def __enter__(self):
        return self

    def __call__(self, func):  # --> used as with context
        @wraps(func)
        def inner(*args, **kwds):
            try:
                with self._recreate_cm():
                    return func(*args, **kwds)
            except Exception as e:
                logger = pb_utils.Logger
                function_stack = traceback.format_exc()
                logger.log_error("function stacks\n{}".format(function_stack))
                local_var = sys.exc_info()[2].tb_next.tb_frame.f_locals
                local_var_pretty = json.dumps(local_var, default=lambda o: str(o)[:10] + "...", indent=4)
                logger.log_error("local variables\n{}".format(local_var_pretty))

        return inner

    def __exit__(self, *exc_details):
        return False

class NumRequestLogger(ContextDecorator):
    """Context Decorator for checking timestamp difference of block"""

    def __enter__(self):
        return self

    def __call__(self, func):  # --> used as with context
        @wraps(func)
        def inner(*args, **kwds):
            logger = pb_utils.Logger
            num_req = 0
            for arg in args[1:]:
                if num_req < len(arg):
                    num_req = len(arg)
            logger.log_info("{} Request coming".format(num_req))

            with self._recreate_cm():
                return func(*args, **kwds)

        return inner

    def __exit__(self, *exc_details):
        return False

async def async_job_distribute(
    loop, thread_pool, task_info_list, task_func, **task_kwargs
):
    """
    distribute each task and await until job ends
    """
    tasks = []
    for per_request_task in task_info_list:
        for task in per_request_task:
            tasks.append(
                asyncio.create_task(
                    async_job_wrapper(
                        loop, thread_pool, partial(task_func, task, **task_kwargs)
                    )
                )
            )
    await asyncio.gather(*tasks)


async def async_job_wrapper(loop, thread_pool, task):
    await loop.run_in_executor(thread_pool, task)

def nd_to_input(arr:np.ndarray):
    # normlize arr
    mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1,1,3])
    arr = (arr / 255 - mean) / std
    return np.transpose(arr, (2, 0, 1))

@ExceptionLogger()
def _read_gcp_img(task, jpeg_wrapper, logging, img_size, transform):
    err_msg = None
    raw = task["raw"]
    try:
        task["img"] = transform(jpeg_wrapper.decode(raw, pixel_format=TJPF_RGB))
        # torch_raw = torch.frombuffer(raw, dtype=torch.uint8)
        # img = decode_image(torch_raw, mode=ImageReadMode.RGB)
        # transformed = transform(img).cpu()
        # task["img"] = transformed.numpy()
    except Exception as nfe:
        logging.log_error(nfe)
        task["img"] = np.empty([1, 3, *img_size])

    if len(task["img"].shape) == 3:
        task["img"] = task["img"][None,...]

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    @ExceptionLogger()
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # logger : triton python backend
        self.logger = pb_utils.Logger

        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args['model_config'])
        self.logger.log_info(json.dumps(args, indent=4))

        # output dtype, dshape
        output_img_config = pb_utils.get_output_config_by_name(
            model_config, "preprocessing_output_img")
        self.output_img_dtype = pb_utils.triton_string_to_numpy(
            output_img_config['data_type'])
        self.output_img_shape = output_img_config['dims'][1:]

        self.model_parameters = {
            k : value_pack["string_value"]
            for k, value_pack in model_config["parameters"].items()
        }

        # # connect GCP bucket
        # if "BUCKET_NAME" in self.model_parameters:
        #     self.storage_client = storage.Client()
        #     adapter = requests.adapters.HTTPAdapter(
        #         pool_connections=self.model_parameters.get("gcp_pool_connection", 8),
        #         pool_maxsize=self.model_parameters.get("gcp_pool_maxsize", 8),
        #         max_retries=self.model_parameters.get("gcp_max_retries", 3),
        #         pool_block=self.model_parameters.get("gcp_pool_block", False),
        #     )
        #     protocol = self.model_parameters.get("gcp_protocol", "http")
        #     self.storage_client._http.mount(f"{protocol}://", adapter)
        #     self.storage_client._http._auth_request.session.mount(
        #         f"{protocol}://", adapter
        #     )
        #     self.bucket = self.storage_client.get_bucket(self.model_parameters["BUCKET_NAME"])
        # else:
        #     # No bucket name given error handling
        #     self.bucket = None
        # self.bucket = None

        # Init event loop / thread pool for async jobs
        self.loop = asyncio.get_event_loop()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(self.model_parameters.get("pool_maxsize", 8))
        self.jpeg_wrapper = TurboJPEG()
        self.transform = nd_to_input

        # Set version info
        self.version = self.model_parameters["VERSION"]

    @ExceptionLogger()
    @NumRequestLogger()
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output_img_dtype = self.output_img_dtype

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.

        task_info_list = []

        for request in requests:
            # Get INPUT0
            request_raw_arr = pb_utils.get_input_tensor_by_name(request, "preprocessing_input").as_numpy()
            per_request_task = []
            for request_raw in request_raw_arr:
                per_request_task.append({
                    "raw": request_raw,
                })
            task_info_list.append(per_request_task)

        self.loop.run_until_complete(
            async_job_distribute(
                self.loop,
                self.thread_pool,
                task_info_list,
                _read_gcp_img,
                jpeg_wrapper=self.jpeg_wrapper,
                logging=self.logger,
                img_size=self.output_img_shape,
                transform=self.transform
            )
        )

        responses = []
        for per_request_task in task_info_list:
            request_imgs = np.concatenate([task["img"] for task in per_request_task])
            request_imgs_tensor = pb_utils.Tensor("preprocessing_output_img",
                                           request_imgs.astype(output_img_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[request_imgs_tensor])
            responses.append(inference_response)
        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    @ExceptionLogger()
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        self.loop.close()