import gradio as gr
from modules import shared


section = ('segment_anything', "Segment Anything")
prefix_id = 'segment_anything_cn_module_'
prefix_label = "CN Preprocessor: "


def getSegmentAnythingModel(sam_model_list):
    if not sam_model_list:
        raise Exception("There are no sam models")

    res : str = shared.opts.data.get(prefix_id + 'sam_model', 'sam_hq_vit_l.pth')
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


def needAutoUnloadModels():
    opt = shared.opts.data.get(prefix_id + "_always_unload_models", 'Enabled')

    if opt == 'Enabled':
        return True
    if opt == 'Disabled':
        return False
    if opt == 'Only SDXL':
        return shared.sd_model.is_sdxl

    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or (shared.sd_model.is_sdxl and shared.cmd_opts.medvram_sdxl)


def getTemplate(sam_model_list):
    if not sam_model_list:
        sam_model_list = ['not found']
    defaultModel = 'sam_hq_vit_l.pth'
    if defaultModel not in sam_model_list:
        defaultModel = sam_model_list[0]

    options = {
        prefix_id + 'sam_model': shared.OptionInfo(
            defaultModel,
            prefix_label + 'segment anything model',
            gr.Dropdown,
            {
                'choices' : sam_model_list,
            },
            section=section,
        ),
        prefix_id + "_always_unload_models": shared.OptionInfo(
            'Enabled',
            prefix_label + 'Always unload models',
            gr.Radio,
            {
                'choices' : ['Automatic', 'Enabled', 'Only SDXL', 'Disabled'],
            },
            section=section,
        ).info("Automatic means enable only for --lowvram and --medvram mode. "),
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
