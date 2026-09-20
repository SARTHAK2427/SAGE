from flash.transport import FlashTransport


def test_role_model_id_suggestions_are_unique_and_role_specific():
    models = ["SAGE Memory Compressor 2B", "Qwen3-VL Flash Vision", "Gemma Flash Controller"]
    transport = FlashTransport()
    assert transport._suggest_model_id("memory", models) == "SAGE Memory Compressor 2B"
    assert transport._suggest_model_id("qwen", models) == "Qwen3-VL Flash Vision"
    assert transport._suggest_model_id("gemma", models) == "Gemma Flash Controller"


def test_ambiguous_model_id_is_not_guessed():
    transport = FlashTransport()
    assert transport._suggest_model_id("gemma", ["gemma-a", "gemma-b"]) is None
