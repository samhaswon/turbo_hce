from typing import Dict, Tuple, Union, Optional, List

from PIL import Image, ImageOps
from PIL.Image import Image as PILImage     # For typing
import onnxruntime as ort
import numpy as np
import cv2


class Session:
    def __init__(
        self,
        model_path: str,
        sess_opts: Optional[ort.SessionOptions] = None,
        providers: Optional[List[str]]=None,
        use_fp16: bool = False,
    ):
        """Initialize an instance of the Session class."""

        self.providers = []
        self.use_fp16 = use_fp16

        if sess_opts is None:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.enable_mem_pattern = True

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        self.inner_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: Image,
        size: Union[Tuple[int, int], None] = None,
        convert_to: str = "RGB"
    ) -> Dict[str, np.ndarray]:
        """
        Normalize and prepare the image for inferencing.
        :param img: The (PIL) Image to repair.
        :param size: The size for inferencing (if not index 2 and 3 of the input shape).
        :param convert_to: The color format to convert the image into (if it even matters).
        :return: The prepared image for ONNX inferencing.
        """
        if size is None:
            size = self.inner_session.get_inputs()[0].shape[2], self.inner_session.get_inputs()[0].shape[3]
        if isinstance(img, np.ndarray):
            im = img
        else:
            im = img.convert(convert_to).resize(size, Image.LANCZOS)

        im_ary = np.array(im).astype(np.float32) / 255.0
        # im_ary = (im_ary - np.min(im_ary)) / (np.max(im_ary) - np.min(im_ary))

        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # im_ary = (im_ary - mean) / std

        tmp_img = im_ary.transpose((2, 0, 1))

        if self.use_fp16:
            tmp_img = tmp_img.astype(np.float16)

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmp_img, 0)
        }

    def predict(self, img: PILImage, size: Union[Tuple[int, int], None] = None, do_sigmoid: bool = False, simple_norm: bool = True, convert_to: str = "RGB") -> np.ndarray:
        """
        Predicts the output mask for the input image using the loaded model.
        :param img: The image to inference on.
        :param size: The prediction size (if not index 2 and 3 of the input shape).
        :return: Prediction mask (numpy.ndarray)
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(img, size, convert_to=convert_to),
        )

        prediction = ort_outs[0][:, 0, :, :]
        if do_sigmoid:
            prediction = 1 / (1 + np.exp(-prediction))

        if not simple_norm:
            mean = prediction.mean(axis=(1, 2), keepdims=True)
            std = prediction.std(axis=(1, 2), keepdims=True)
            prediction[0, ...] = (prediction[0, ...] - mean[0]) / (std[0] + 1E-8)
        else:
            ma = np.max(prediction)
            mi = np.min(prediction)
            # print(f"{ma = }; {mi = }; {ma - mi = }; {np.sum(prediction) = }")
            if (ma != mi and (ma != 1.0 and mi != 0.0)) and mi < 0.981:
                prediction = (prediction - mi) / (ma - mi + 1E-8)

        prediction = np.squeeze(prediction)

        mask = cv2.resize(
            (prediction * 255).astype("uint8"),
            img.size,
            interpolation=cv2.INTER_LANCZOS4)

        return mask

    @staticmethod
    def fix_image_orientation(img: PILImage) -> PILImage:
        """
        Fix the orientation of the image based on its EXIF data.
        :param img: The image to be fixed.
        :returns: PILImage: The fixed image.
        """
        return ImageOps.exif_transpose(img)

    @staticmethod
    def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
        """
        Perform a simple cutout operation on an image using a mask.
        :param img: Base image to cut out from.
        :param mask: Mask to use to cut out from the base image. A.k.a., the alpha channel.
        """
        empty = Image.new("RGBA", img.size, 0)
        cutout = Image.composite(img, empty, mask)
        return cutout

    @staticmethod
    def post_process(mask: np.ndarray) -> np.ndarray:
        """
        Morphs and blurs the mask to make it a bit better (generally speaking).
        :param mask: The mask to post-process.
        :return: The post-processed mask.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.GaussianBlur(mask, (9, 9), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
        return mask

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False,
            do_sigmoid: bool = False,
            simple_norm: bool = True,
    ) -> PILImage:
        """
        Segment an input image.
        :param do_sigmoid: Whether to perform sigmoid or not on the output.
        :param img: Image to segment.
        :param size: The inferencing size (if not index 2 and 3 of the input shape).
        :param mask_only: Whether to return only the mask or a cutout.
        :return: Segmented image or mask
        """

        # Fix image orientation
        img = self.fix_image_orientation(img)

        mask = self.predict(img, size, do_sigmoid=do_sigmoid, simple_norm=simple_norm)

        mask = Image.fromarray(self.post_process(mask), mode="L")
        if mask_only:
            cutout = mask
        else:
            cutout = self.naive_cutout(img, mask)

        return cutout
