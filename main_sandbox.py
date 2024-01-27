import pytorch_lightning.utilities.model_summary.model_summary

from modules.lit_iadb_colorizer import LitIADBColorizer

model = LitIADBColorizer(
        n_features_list=(64, 128, 256, 512, 1024),
        use_attention_list=(False, False, False, False, False)
    )


print(pytorch_lightning.utilities.model_summary.summarize(model))