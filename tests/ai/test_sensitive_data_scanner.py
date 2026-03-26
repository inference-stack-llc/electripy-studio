"""Tests for SensitiveDataScanner."""

from __future__ import annotations

import re

from electripy.ai.sensitive_data_scanner import (
    SensitiveDataScanner,
    SensitivePattern,
    scan_text,
)


class TestScanText:
    def test_email_detected(self) -> None:
        matches = scan_text("Contact alice@example.com for details")
        categories = [m.category for m in matches]
        assert "email" in categories
        assert any(m.matched_text == "alice@example.com" for m in matches)

    def test_phone_detected(self) -> None:
        matches = scan_text("Call me at 555-123-4567")
        categories = [m.category for m in matches]
        assert "phone_us" in categories

    def test_ssn_detected(self) -> None:
        matches = scan_text("SSN: 123-45-6789")
        categories = [m.category for m in matches]
        assert "ssn" in categories
        assert any(m.matched_text == "123-45-6789" for m in matches)

    def test_openai_key_detected(self) -> None:
        matches = scan_text("Key: sk-abc123def456ghi789jkl012mno")
        categories = [m.category for m in matches]
        assert "api_key_openai" in categories

    def test_anthropic_key_detected(self) -> None:
        matches = scan_text("Key: sk-ant-abc123def456ghi789jkl012mno")
        categories = [m.category for m in matches]
        assert "api_key_anthropic" in categories

    def test_aws_key_detected(self) -> None:
        matches = scan_text("AWS key: AKIA1234567890ABCDEF")
        categories = [m.category for m in matches]
        assert "aws_access_key" in categories

    def test_ipv4_detected(self) -> None:
        matches = scan_text("Server at 192.168.1.100")
        categories = [m.category for m in matches]
        assert "ipv4" in categories

    def test_no_sensitive_data(self) -> None:
        matches = scan_text("This is a perfectly safe sentence.")
        assert matches == []

    def test_multiple_findings(self) -> None:
        text = "Email alice@corp.io, SSN 123-45-6789, call 555-111-2222"
        matches = scan_text(text)
        assert len(matches) >= 3

    def test_sorted_by_position(self) -> None:
        text = "SSN 123-45-6789 and email alice@corp.io"
        matches = scan_text(text)
        positions = [m.start for m in matches]
        assert positions == sorted(positions)

    def test_match_fields(self) -> None:
        matches = scan_text("alice@test.com")
        assert len(matches) >= 1
        m = next(m for m in matches if m.category == "email")
        assert m.matched_text == "alice@test.com"
        assert m.start == 0
        assert m.end == 14


class TestSensitiveDataScanner:
    def test_custom_pattern(self) -> None:
        scanner = SensitiveDataScanner(
            include_builtins=False,
            patterns=[
                SensitivePattern(
                    category="employee_id",
                    pattern=re.compile(r"\bEMP-\d{6}\b"),
                    description="Employee ID",
                ),
            ],
        )
        matches = scanner.scan("Employee EMP-123456 is active")
        assert len(matches) == 1
        assert matches[0].category == "employee_id"

    def test_add_pattern(self) -> None:
        scanner = SensitiveDataScanner(include_builtins=False)
        scanner.add_pattern(
            SensitivePattern(
                category="custom",
                pattern=re.compile(r"\bSECRET\b"),
            )
        )
        matches = scanner.scan("This is SECRET data")
        assert len(matches) == 1

    def test_has_sensitive_data_true(self) -> None:
        scanner = SensitiveDataScanner()
        assert scanner.has_sensitive_data("email: test@test.com") is True

    def test_has_sensitive_data_false(self) -> None:
        scanner = SensitiveDataScanner()
        assert scanner.has_sensitive_data("nothing sensitive here") is False

    def test_no_builtins(self) -> None:
        scanner = SensitiveDataScanner(include_builtins=False)
        matches = scanner.scan("alice@test.com 123-45-6789")
        assert matches == []

    def test_categories(self) -> None:
        scanner = SensitiveDataScanner()
        cats = scanner.categories
        assert "email" in cats
        assert "ssn" in cats
        assert "api_key_openai" in cats
