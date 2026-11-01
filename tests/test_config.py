from app.config import GPU_BUDGET_GB, PROFILES, Budgets, Settings, estimate_memory_gb


def test_every_profile_fits_its_context_and_the_24gb_budget():
    for name, prof in PROFILES.items():
        s = Settings(profile=name).resolved()
        s.budgets.check()
        est = estimate_memory_gb(s.gen_model, s.embed_model, s.budgets.num_ctx, prof.kv_cache_type)
        assert 0 < est <= GPU_BUDGET_GB, (name, est)


def test_auto_detect_prefers_a_pulled_model():
    s = Settings(profile="balanced").resolved(["llama3.1:8b", "nomic-embed-text:latest"])
    assert s.gen_model == "llama3.1:8b"
    s = Settings(profile="balanced").resolved(["qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b"])
    assert s.gen_model == "qwen2.5:7b-instruct"


def test_explicit_model_and_context_override_profile():
    s = Settings(gen_model="qwen2.5:14b-instruct", num_ctx=8192).resolved(["qwen2.5:7b-instruct"])
    assert s.gen_model == "qwen2.5:14b-instruct" and s.budgets.num_ctx == 8192


def test_scaled_budgets_leave_a_safety_margin():
    for ctx in (4096, 8192, 16384, 32768):
        b = Budgets.for_ctx(ctx)
        b.check()
        assert b.sources > 0
