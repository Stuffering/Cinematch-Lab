import pytest

from cinematch.recommendation_agent import (
    RecommendationRequest,
    build_recommendation_request,
    choose_recommendation_mode,
    list_recommendation_modes,
    validate_recommendation_mode,
)


def test_list_recommendation_modes_returns_stable_ui_order() -> None:
    modes = list_recommendation_modes()

    assert modes == ["hybrid", "item_cf", "content"]


def test_validate_recommendation_mode_normalizes_supported_mode() -> None:
    assert validate_recommendation_mode(" HYBRID ") == "hybrid"

    with pytest.raises(ValueError, match="Unsupported recommendation mode"):
        validate_recommendation_mode("neural")


@pytest.mark.skip(reason="Stage 12 learning step: build recommendation request")
def test_build_recommendation_request_returns_validated_request() -> None:
    request = build_recommendation_request(user_id=1, mode="CONTENT", n=5)

    assert request == RecommendationRequest(user_id=1, mode="content", n=5)


@pytest.mark.skip(reason="Stage 12 learning step: choose recommendation mode")
def test_choose_recommendation_mode_uses_requested_mode_when_explicit() -> None:
    mode = choose_recommendation_mode(user_rating_count=100, requested_mode="item_cf")

    assert mode == "item_cf"
