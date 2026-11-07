"""Test LLM API wrappers (parse prompt, package exports)."""

from unittest.mock import MagicMock, patch

import pytest


class TestOpenAIWrapperParse:
    def test_parse_prompt_plain(self):
        from vietnamese_ai.llm.wrappers import OpenAIWrapper

        w = object.__new__(OpenAIWrapper)
        msgs = OpenAIWrapper._parse_prompt(w, "Xin chào Việt Nam")
        assert msgs == [{"role": "user", "content": "Xin chào Việt Nam"}]

    def test_parse_prompt_roles(self):
        from vietnamese_ai.llm.wrappers import OpenAIWrapper

        w = object.__new__(OpenAIWrapper)
        prompt = "Hệ thống: Bạn là trợ lý\nNgười dùng: Chào\nTrợ lý: Xin chào"
        msgs = OpenAIWrapper._parse_prompt(w, prompt)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"]

    def test_sinh_van_ban_voi_client_mock(self):
        from vietnamese_ai.llm.wrappers import OpenAIWrapper

        w = object.__new__(OpenAIWrapper)
        w.model = "gpt-test"
        w.max_tokens = 64
        w.temperature = 0.5
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Xin chào"))]
        w.client = MagicMock()
        w.client.chat.completions.create.return_value = mock_resp

        out = w.sinh_van_ban("Người dùng: Hi")
        assert out == "Xin chào"
        w.client.chat.completions.create.assert_called_once()


class TestGeminiWrapper:
    def test_thieu_package(self):
        from vietnamese_ai.llm import wrappers

        real_import = __import__

        def blocker(name, *args, **kwargs):
            if name.startswith("google"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocker):
            with pytest.raises(ImportError, match="google-generativeai"):
                wrappers.GeminiWrapper(api_key="fake")


class TestLLMExports:
    def test_export_tu_package(self):
        from vietnamese_ai.llm import GeminiWrapper, OpenAIWrapper, VietnameseLLM

        assert OpenAIWrapper is not None
        assert GeminiWrapper is not None
        assert VietnameseLLM is not None
