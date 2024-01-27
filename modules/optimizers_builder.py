import adan_pytorch


def build_optimizer(params, config):
    if config["name"] == "Adan":
        return adan_pytorch.Adan(params, lr=config["lr"])