from PIL import Image
from scripts.supported_preprocessor import Preprocessor
from scripts.utils import resize_image_with_pad
from scripts import sam
from modules import shared
from cn_sam_module.tools import convertImageIntoPILFormat, convertIntoCNImageFormat
from cn_sam_module.options import getTemplate, getAutoSamOptions, getSegmentAnythingModel


def processAutoSegmentAnything(image: Image.Image):
    sam_model_name = getSegmentAnythingModel(sam.sam_model_list)
    nones = ['random', None, None, None, None, None]
    auto_sam_config = getAutoSamOptions()
    result = sam.cnet_seg(sam_model_name, image, *nones, *auto_sam_config)
    print(result[1])
    return result[0][1]



class PreprocessorSegmentAnything(Preprocessor):
    def __init__(self):
        super().__init__(name="segment_anything")
        self.tags = ["Segmentation"]

    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        **kwargs
    ):
        img, remove_pad = resize_image_with_pad(input_image, resolution)
        img = convertImageIntoPILFormat(img)

        result = processAutoSegmentAnything(img)

        result = convertIntoCNImageFormat(result)
        result = remove_pad(result)
        return result


Preprocessor.add_supported_preprocessor(PreprocessorSegmentAnything())
shared.options_templates.update(getTemplate(sam.sam_model_list))
