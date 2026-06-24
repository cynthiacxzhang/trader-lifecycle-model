from src.config import settings


def test_settings_loads():
    assert settings.annual_mgmt_fee == 0.0025
    assert settings.monthly_discount_rate == 0.005
    assert settings.n_users == 5000
    assert settings.n_days == 730
    assert settings.random_seed == 42
