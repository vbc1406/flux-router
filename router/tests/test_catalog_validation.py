from scripts.validate_model_catalog import DOCUMENTED_PROVIDER_IDS, load_models, validate_local


def test_catalog_passes_deterministic_validation():
    assert validate_local(load_models()) == []


def test_every_documented_dispatch_override_is_present():
    models = {model["model_id"]: model for model in load_models()}
    for model_id, provider_model_id in DOCUMENTED_PROVIDER_IDS.items():
        assert models[model_id]["provider_model_id"] == provider_model_id
