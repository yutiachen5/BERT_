from sklearn.neural_network import MLPRegressor


def MLPRegression(logger,
                  variant: str,
                  **kwargs):
    if variant == "MLP":
        logger.info("Loading MLP Regression model")
        model = MLPRegressor(**kwargs)
    return model
