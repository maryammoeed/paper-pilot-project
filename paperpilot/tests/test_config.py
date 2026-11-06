import os

from app.config import load_dotenv


def test_dotenv_loads_values_but_real_env_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nPP_TEST_A=from_file\nPP_TEST_B='quoted'  # note\nPP_TEST_C=from_file\n")
    monkeypatch.delenv("PP_TEST_A", raising=False)
    monkeypatch.delenv("PP_TEST_B", raising=False)
    monkeypatch.setenv("PP_TEST_C", "from_env")
    load_dotenv(env)
    assert os.environ["PP_TEST_A"] == "from_file"
    assert os.environ["PP_TEST_B"] == "quoted"
    assert os.environ["PP_TEST_C"] == "from_env"
    for k in ("PP_TEST_A", "PP_TEST_B"):
        monkeypatch.delenv(k)
