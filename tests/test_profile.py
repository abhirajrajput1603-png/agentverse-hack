import pytest
from xray.profile import UserRiskProfile


def test_user_risk_profile_defaults():
    profile = UserRiskProfile()
    assert profile.name == "Everyday Investor"
    assert profile.horizon_years == 3.0
    assert profile.max_single_stock_limit == 0.20
    assert profile.max_tolerable_loss_pct == 0.20
    assert profile.experience_level == "Intermediate"


def test_user_risk_profile_serialization():
    data = {
        "name": "Liam",
        "horizon_years": 3.0,
        "max_tolerable_loss_pct": 0.15,
        "max_single_stock_limit": 0.20,
        "objective": "House deposit preservation",
        "experience_level": "Beginner",
        "financial_milestone": "£60,000 in 3 years",
    }
    p = UserRiskProfile.from_dict(data)
    assert p.name == "Liam"
    assert p.max_tolerable_loss_pct == 0.15
    assert p.financial_milestone == "£60,000 in 3 years"

    d = p.to_dict()
    assert d["name"] == "Liam"
    assert d["objective"] == "House deposit preservation"
