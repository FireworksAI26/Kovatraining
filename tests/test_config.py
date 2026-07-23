from dataclasses import replace

import pytest

from kova_training.config import CampaignConfig, ConfigurationError


def test_default_campaign_is_valid_and_smoke_is_capped() -> None:
    config = CampaignConfig()
    config.validate()
    assert config.maximum_smoke_cost < config.budget.smoke_cap


def test_unsafe_smoke_ttl_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="hard cap"):
        replace(CampaignConfig(), smoke_ttl_minutes=400).validate()


def test_mutable_model_revision_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="immutable"):
        replace(CampaignConfig(), model_revision="main").validate()
