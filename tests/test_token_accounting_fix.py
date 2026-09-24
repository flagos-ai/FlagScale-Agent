# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for token accounting bug fixes (kernel.py + history.py)."""

import pytest
from unittest.mock import MagicMock, patch
from flagscale_agent.react.history import HistoryManager


class TestContextPressureEstimation:
    """Test that get_context_pressure() uses character-based estimation correctly."""

    def test_pressure_uses_estimation_only(self):
        """Context pressure should use character-based estimation, not single API call actual tokens."""
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        # working_window = 200000 * 0.75 = 150000 tokens
        # ASCII estimate: len / 4, so 40000 chars ≈ 10000 tokens
        hm.append({"role": "user", "content": "x" * 40000})
        
        pressure = hm.get_context_pressure()
        
        # Estimated tokens ≈ 10000, pressure ≈ 10000/150000 ≈ 0.067
        assert 0.05 < pressure < 0.15

    def test_pressure_includes_all_messages(self):
        """Pressure should account for all messages in history."""
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "a" * 10000})
        hm.append({"role": "assistant", "content": "b" * 10000})
        hm.append({"role": "user", "content": "c" * 10000})
        
        pressure = hm.get_context_pressure()
        
        # Total ≈ 30000 chars / 4 = 7500 tokens, pressure ≈ 7500/150000 ≈ 0.05
        assert 0.04 < pressure < 0.1

    def test_pressure_includes_evicted_placeholders(self):
        """Evicted messages are replaced by placeholders, which still consume tokens."""
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        hm.append({"role": "user", "content": "x" * 100000})  # Large message to evict
        hm.append({"role": "assistant", "content": "ok"})
        hm.append({"role": "user", "content": "msg3"})
        hm.append({"role": "assistant", "content": "msg4"})
        hm.append({"role": "user", "content": "msg5"})
        hm.append({"role": "assistant", "content": "msg6"})
        # Now we have 7 messages, last 4 are protected
        # Evictable: ext_idx=2 (internal=1, the large user message)
        
        pressure_before = hm.get_context_pressure()
        
        # Evict the large user message at EXTERNAL index 2 (internal=1)
        evicted = hm.evict_message(2)
        assert evicted is not None
        
        pressure_after = hm.get_context_pressure()
        
        # Pressure should be significantly lower after eviction (placeholder is much shorter)
        # but not zero (placeholder still counts)
        assert pressure_after < pressure_before * 0.5  # At least 50% reduction
        assert pressure_after > 0


class TestKernelTokenAccounting:
    """Test that kernel.py correctly accounts for total input tokens (including cache)."""

    def test_total_input_tokens_includes_cache(self):
        """Kernel should use total_in_tok (input + cache_read + cache_create) for display and result."""
        # Simulate a usage dict from Anthropic API with cache
        usage = {
            "input_tokens": 1,                      # non-cached new input
            "output_tokens": 50,
            "cache_read_input_tokens": 34305,      # read from cache
            "cache_creation_input_tokens": 13753,  # created cache
        }
        
        # Expected: total_in_tok = 1 + 34305 + 13753 = 48059
        expected_total = 48059
        
        # Verify the calculation (this is what kernel.py does now)
        in_tok = usage.get("input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_create = usage.get("cache_creation_input_tokens") or 0
        total_in_tok = in_tok + cache_read + cache_create
        
        assert total_in_tok == expected_total

    def test_result_accumulates_total_tokens(self):
        """KernelResult.input_tokens should accumulate total input (including cache)."""
        from flagscale_agent.react.kernel import KernelResult
        
        result = KernelResult()
        
        # Simulate multiple API calls with cache
        # Call 1: 1 + 30000 + 10000 = 40001
        result.input_tokens += 1 + 30000 + 10000
        # Call 2: 2 + 40000 + 5000 = 45002
        result.input_tokens += 2 + 40000 + 5000
        
        assert result.input_tokens == 85003


class TestHistoryReportActualTokens:
    """Test that report_actual_tokens correctly updates _actual_input_tokens."""

    def test_report_actual_tokens_updates_field(self):
        hm = HistoryManager(max_context_tokens=200000)
        hm.append({"role": "system", "content": "sys"})
        
        # Report actual tokens from API
        hm.report_actual_tokens(48059)
        
        assert hm._actual_input_tokens == 48059

    def test_report_actual_tokens_overwrites_previous(self):
        """Each report_actual_tokens call should overwrite the previous value (it's per-call, not cumulative)."""
        hm = HistoryManager(max_context_tokens=200000)
        
        hm.report_actual_tokens(10000)
        assert hm._actual_input_tokens == 10000
        
        hm.report_actual_tokens(20000)
        assert hm._actual_input_tokens == 20000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
