from modules.denoising_unet import DenoisingUNet


def build_model(inner_model_config):
    if inner_model_config["module_name"] == "DenoisingUNet":
        return DenoisingUNet(
            n_features_list=inner_model_config.get("n_features_list", (128, 256, 256)),
            use_attention_list=inner_model_config.get("use_attention_list", (False, True, True)),
            embedding_dim=inner_model_config.get("embedding_dim", 256)
        )

