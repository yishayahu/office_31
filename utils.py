def freeze_model(model, exclude_layers):
    for i,(name, param) in enumerate(model.named_parameters()):
        requires_grad = False

        if i  in exclude_layers:
            requires_grad = True
        param.requires_grad = requires_grad
