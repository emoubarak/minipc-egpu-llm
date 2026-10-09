#!/usr/bin/env bash
# Tests for bin/ai-auto that need no GPU and no real model.
# A GGUF header is attacker-controlled input: none of its values may reach the shell as code.
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
AI="$ROOT/bin/ai-auto"
MK="$ROOT/tests/make_gguf.py"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 1
export AI_MODELS_DIR="$WORK/models" LLAMA_DIR="$WORK/no-llama"
mkdir -p "$AI_MODELS_DIR"

fails=0
pass() { echo "ok   $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }

# 1. Payloads in every field the script reads. Each one would create a file if evaluated.
payload_int='BASH_VERSINFO[$(touch pwned_int)]'
payload_str='x$(touch pwned_arch)'
python3 -I "$MK" "$AI_MODELS_DIR/evil.gguf" \
    "general.architecture|str|$payload_str" \
    "evil.expert_count|str|$payload_int" \
    "evil.block_count|str|$payload_int" \
    "evil.embedding_length|str|1\nGPU=\$(touch pwned_nl)" \
    "evil.expert_feed_forward_length|str|512=\$(touch pwned_eq)" \
    "evil.nextn_predict_layers|str|$payload_int" \
    "evil.context_length|bool|true" \
    "evil.head_count|f32|1e30" \
    "evil\nGPU=\$(touch pwned_key).head_count_kv|u32|8"
"$AI" info "$AI_MODELS_DIR/evil.gguf" >/dev/null 2>&1
"$AI" list >/dev/null 2>&1
AI_GPU=igpu "$AI" info "$AI_MODELS_DIR/evil.gguf" >/dev/null 2>&1
if compgen -G "$WORK/pwned*" >/dev/null; then
    fail "crafted GGUF executed code: $(ls "$WORK" | grep pwned | tr '\n' ' ')"
else
    pass "crafted GGUF: no code executed"
fi

# 2. The hostile values are dropped, not passed on.
out=$(AI_GPU=igpu "$AI" info "$AI_MODELS_DIR/evil.gguf" 2>&1)
if grep -q 'architecture : unknown' <<<"$out" && grep -q 'type         : dense, 32 layers' <<<"$out"; then
    pass "crafted GGUF: values replaced by defaults"
else
    fail "crafted GGUF: unexpected output"; echo "$out" | sed 's/^/     /'
fi

# 3. A valid MoE header with an MTP head is read correctly.
python3 -I "$MK" "$AI_MODELS_DIR/moe.gguf" \
    "general.architecture|str|qwen35moe" "qwen35moe.block_count|u32|41" \
    "qwen35moe.expert_count|u32|256" "qwen35moe.expert_used_count|u32|8" \
    "qwen35moe.embedding_length|u32|2048" "qwen35moe.expert_feed_forward_length|u32|512" \
    "qwen35moe.nextn_predict_layers|u32|1" "qwen35moe.context_length|u32|262144" \
    "qwen35moe.attention.head_count|u32|16" "qwen35moe.attention.head_count_kv|u32|2"
out=$(AI_GPU=igpu "$AI" info "$AI_MODELS_DIR/moe.gguf" 2>&1)
if grep -q 'MoE, 256 experts, 41 layers' <<<"$out" && grep -q 'draft-mtp' <<<"$out"; then
    pass "valid MoE + MTP header parsed"
else
    fail "valid MoE + MTP header"; echo "$out" | sed 's/^/     /'
fi

# 4. Paths with spaces stay one argument.
mkdir -p "$WORK/dir with space"
cp "$AI_MODELS_DIR/moe.gguf" "$WORK/dir with space/model.gguf"
python3 -I "$MK" "$WORK/dir with space/mmproj-x.gguf" "general.architecture|str|clip"
out=$(AI_GPU=igpu AI_MMPROJ="$WORK/dir with space/mmproj-x.gguf" "$AI" info "$WORK/dir with space/model.gguf" 2>&1)
if grep -qF -- '--mmproj '"$(printf '%q' "$WORK/dir with space/mmproj-x.gguf")" <<<"$out"; then
    pass "path with spaces quoted in the printed command"
else
    fail "path with spaces"; echo "$out" | sed 's/^/     /'
fi

# 5. Errors: one clear message, non-zero exit.
out=$("$AI" info "$WORK/missing.gguf" 2>&1); rc=$?
if (( rc != 0 )) && [[ $(grep -c '✗' <<<"$out") -eq 1 ]]; then pass "missing model: one error"; else fail "missing model: rc=$rc"; echo "$out"; fi
printf 'not a gguf' > "$WORK/bad.gguf"
out=$("$AI" info "$WORK/bad.gguf" 2>&1); rc=$?
if (( rc != 0 )) && [[ $(grep -c '✗' <<<"$out") -eq 1 ]] && grep -q 'not a GGUF' <<<"$out"; then
    pass "not a GGUF: one error"
else
    fail "not a GGUF: rc=$rc"; echo "$out"
fi
out=$(AI_NCMOE='1+$(touch pwned_env)' AI_GPU=igpu "$AI" info "$AI_MODELS_DIR/moe.gguf" 2>&1); rc=$?
if (( rc != 0 )) && [[ ! -e "$WORK/pwned_env" ]]; then pass "non-numeric AI_NCMOE rejected"; else fail "AI_NCMOE: rc=$rc"; fi

# 6. Deeply nested arrays: a clean error, not a Python traceback.
python3 -I "$MK" "$WORK/deep.gguf" "general.architecture|str|x" "x.junk|nested|5000"
out=$("$AI" info "$WORK/deep.gguf" 2>&1); rc=$?
if (( rc != 0 )) && ! grep -q Traceback <<<"$out" && grep -q 'could not read GGUF metadata' <<<"$out"; then
    pass "deeply nested arrays: clean error"
else
    fail "deeply nested arrays: rc=$rc"; echo "$out" | tail -3
fi

# 7. Drafters are listed as such.
python3 -I "$MK" "$AI_MODELS_DIR/drafter.gguf" "general.architecture|str|dflash" "dflash.block_count|u32|6"
if "$AI" list 2>/dev/null | grep -E 'drafter\.gguf' | grep -q ' draft '; then pass "drafter listed as draft"; else fail "drafter in list"; fi

echo
(( fails == 0 )) && echo "all tests passed" || { echo "$fails test(s) failed"; exit 1; }
