import torch

def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"

def patch_gptq(config):
    do_gptq_patching = False
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    do_gptq_patching = quantization_config and quantization_config["quant_method"] == "gptq"
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    if do_gptq_patching:
        torch.set_default_dtype(torch.float32)
        torch.cuda.is_available = lambda: True

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model