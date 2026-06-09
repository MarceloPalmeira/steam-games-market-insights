from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
PLACEHOLDER_KEY = "coloque_sua_chave_aqui"


def _load_api_settings() -> None:
    load_dotenv(ENV_PATH, override=True)


def _get_api_key() -> str:
    _load_api_settings()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key == PLACEHOLDER_KEY:
        return ""
    return api_key


def _candidate_models() -> list[str]:
    _load_api_settings()
    primary_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    env_fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").strip()
    fallback_models = (
        [model.strip() for model in env_fallbacks.split(",") if model.strip()]
        if env_fallbacks
        else FALLBACK_GEMINI_MODELS
    )

    candidates = [primary_model, *fallback_models]
    return list(dict.fromkeys(model for model in candidates if model))


def _is_model_fallback_error(exc: Exception) -> bool:
    text = str(exc)
    retryable_markers = [
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "NOT_FOUND",
        "not found",
        "not supported",
        "not available",
        "permission denied for model",
    ]
    blocked_markers = ["API_KEY_INVALID", "API key not valid", "GEMINI_API_KEY nao configurada"]
    if any(marker in text for marker in blocked_markers):
        return False
    return any(marker in text for marker in retryable_markers)


def gemini_api_key_available() -> bool:
    return bool(_get_api_key())


def generate_gemini_report(payload: dict[str, Any], output_path: Path) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao configurada. Crie um arquivo .env baseado em .env.example.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Pacote google-genai nao instalado. Rode: python3 -m pip install -r requirements.txt") from exc

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(payload)
    errors: list[str] = []

    for model_name in _candidate_models():
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            report = getattr(response, "text", "") or ""
            if not report.strip():
                raise RuntimeError("O Gemini retornou uma resposta vazia.")

            report = f"Modelo Gemini usado: `{model_name}`\n\n{report}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            return report
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            if not _is_model_fallback_error(exc):
                raise

    error_summary = "\n".join(errors)
    raise RuntimeError(f"Nenhum modelo Gemini de fallback respondeu com sucesso. Tentativas:\n{error_summary}")


def _build_prompt(payload: dict[str, Any]) -> str:
    safe_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
Voce e um consultor de Ciencia de Dados ajudando um desenvolvedor indie/publisher.

Escreva um relatorio executivo em Markdown, em portugues do Brasil, sobre o projeto
Steam Games Market Insights. Use apenas os dados agregados abaixo. Nao invente
receita real e nao trate associacoes como causalidade.

O relatorio deve conter:
- resumo do problema de negocio;
- definicao curta da variavel-alvo success_commercial como proxy;
- modelos avaliados e melhor modelo;
- interpretacao objetiva das metricas;
- principais features quando disponiveis;
- pelo menos 3 recomendacoes acionaveis;
- limitacoes e cuidados de interpretacao.

Dados agregados permitidos:
{safe_payload}
""".strip()
