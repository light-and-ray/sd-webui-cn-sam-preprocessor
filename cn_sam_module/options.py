import gradio as gr
from modules import shared


section = ('segment_anything', "Segment Anything")
prefix_id = 'segment_anything_cn_module_'
prefix_label = "CN Module: "


def getSegmentAnythingModel(sam_model_list):
    if not sam_model_list:
        raise Exception("There are no sam models")
    if 'sam_hq_vit_l.pth' in sam_model_list:
        default = 'sam_hq_vit_l.pth'
    else:
        default = sam_model_list[0]
    res : str = shared.opts.data.get(prefix_id + 'sam_model', default)
    if res not in sam_model_list:
        res = sam_model_list[0]
    return res


def getAutoSamOptions():
    res = []
    def append(id, default):
        value = shared.opts.data.get(prefix_id + id, default)
        if value % 1 == 0:
            value = int(value)
        res.append(value)
    append('points_per_side', 32)
    append('points_per_batch', 64)
    append('pred_iou_thresh', 0.88)
    append('stability_score_thresh', 0.95)
    append('stability_score_offset', 1)
    append('box_nms_thresh', 0.7)
    append('crop_n_layers', 0)
    append('crop_nms_thresh', 0.7)
    append('crop_overlap_ratio', 512/1500)
    append('crop_n_points_downscale_factor', 1)
    append('min_mask_region_area', 0)
    return res


def getTemplate(sam_model_list):
    if not sam_model_list:
        sam_model_list = ['not found']

    options = {
        prefix_id + 'sam_model': shared.OptionInfo(
            sam_model_list[0],
            prefix_label + 'segment anything model',
            gr.Radio,
            {
                'choices' : sam_model_list,
            },
            section=section,
        )
    }

    def addNumberOption(label: str, value):
        args = {}
        if value % 1 == 0:
            args['step'] = 1
        options[prefix_id + label] = shared.OptionInfo(
            value,
            prefix_label + label,
            gr.Number,
            args,
            section=section,
        )

    addNumberOption('points_per_side', 32)
    addNumberOption('points_per_batch', 64)
    addNumberOption('pred_iou_thresh', 0.88)
    addNumberOption('stability_score_thresh', 0.95)
    addNumberOption('stability_score_offset', 1)
    addNumberOption('box_nms_thresh', 0.7)
    addNumberOption('crop_n_layers', 0)
    addNumberOption('crop_nms_thresh', 0.7)
    addNumberOption('crop_overlap_ratio', 512/1500)
    addNumberOption('crop_n_points_downscale_factor', 1)
    addNumberOption('min_mask_region_area', 0)

    return shared.options_section(section, options)
