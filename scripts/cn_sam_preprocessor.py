from PIL import Image
from torch.cuda import OutOfMemoryError
from scripts.supported_preprocessor import Preprocessor
from scripts.utils import resize_image_with_pad
from scripts import sam
from modules import shared, sd_models, devices
from cn_sam_preprocessor.tools import convertImageIntoPILFormat, convertIntoCNImageFormat
from cn_sam_preprocessor.options import (getTemplate, getAutoSamOptions,
    getSegmentAnythingModel, needAutoUnloadModels, avoidOOM,
)


def unloadSAM():
    sam.clear_cache()
    devices.torch_gc()


def processAutoSegmentAnything(image: Image.Image):
    sam_model_name = getSegmentAnythingModel(sam.sam_model_list)
    nones = ['random', None, None, None, None, None]
    auto_sam_config = getAutoSamOptions()
    print(sam_model_name, auto_sam_config)
    result = sam.cnet_seg(sam_model_name, image, *nones, *auto_sam_config)
    print(result[1])
    return result[0][1]



def processAutoSegmentAnything_cpu(image: Image.Image):
    print('Using cpu for auto segmentation')
    oldDevice = devices.device
    oldSamDevice = sam.sam_device
    devices.device = 'cpu'
    sam.sam_device = 'cpu'
    try:
        return processAutoSegmentAnything(image)
    finally:
        devices.device = oldDevice
        sam.sam_device = oldSamDevice



def processAutoSegmentAnything_avoidOOM(image: Image.Image):
    try:
        result = processAutoSegmentAnything(image)
    except OutOfMemoryError:
        print("\nOut of Memory. Unload Stable Diffusion\n")
        unloadSAM()
        sd_models.unload_model_weights()
        try:
            result = processAutoSegmentAnything(image)
        except OutOfMemoryError:
            print("\nOut of Memory. Use CPU\n")
            unloadSAM()
            result = processAutoSegmentAnything_cpu(image)
    return result



class PreprocessorSegmentAnything(Preprocessor):
    NAME = "segment_anything"

    def __init__(self):
        super().__init__(name=self.NAME)
        self.tags = ["Segmentation"]

    def unload(self) -> bool:
        """@Override"""
        unloadSAM()
        return True

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


        if avoidOOM():
            result = processAutoSegmentAnything_avoidOOM(img)
        else:
            result = processAutoSegmentAnything(img)
        if needAutoUnloadModels(): unloadSAM()

        result = convertIntoCNImageFormat(result)
        result = remove_pad(result)
        return result


shared.options_templates.update(getTemplate(sam.sam_model_list))
if not Preprocessor.get_preprocessor(PreprocessorSegmentAnything.NAME):
    Preprocessor.add_supported_preprocessor(PreprocessorSegmentAnything())
