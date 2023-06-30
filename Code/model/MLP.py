from sklearn.neural_network import MLPRegressor


def MLPRegression(logger,
                  variant: str,
                  **kwargs):
    if variant == "MLP":
        logger.info("Loading MLP Regression model.")
        model = MLPRegressor(**kwargs)
    else:
        raise ValueError(f"Unrecognized variant: {variant}.")
    return model
