import os
import time
import hmac
import hashlib
import base64
import json
from typing import Any, Dict


def _b64url_encode(data: bytes) -> str:
	return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
	padding = '=' * ((4 - len(data) % 4) % 4)
	return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
	salt = os.getenv("JWT_PASS").encode("utf-8")
	dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
	return _b64url_encode(dk)


def verify_password(password: str, hashed: str) -> bool:
	return hmac.compare_digest(hash_password(password), hashed)


def create_jwt(payload: Dict[str, Any], expires_in_seconds: int = 3600) -> str:
	secret = os.getenv("JWT_SECRET").encode("utf-8")
	alg = "HS256"
	header = {"alg": alg, "typ": "JWT"}
	claims = dict(payload)
	claims["exp"] = int(time.time()) + expires_in_seconds

	segments = [
		_b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
		_b64url_encode(json.dumps(claims, separators=(",", ":")).encode("utf-8")),
	]
	signing_input = ".".join(segments).encode("ascii")
	signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
	segments.append(_b64url_encode(signature))
	return ".".join(segments)


def verify_jwt(token: str) -> Dict[str, Any] | None:
	try:
		secret = os.getenv("JWT_SECRET").encode("utf-8")
		header_b64, payload_b64, sig_b64 = token.split(".")
		signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
		expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()
		if not hmac.compare_digest(expected_sig, _b64url_decode(sig_b64)):
			return None
		claims = json.loads(_b64url_decode(payload_b64))
		if int(time.time()) >= int(claims.get("exp", 0)):
			return None
		return claims
	except Exception:
		return None


