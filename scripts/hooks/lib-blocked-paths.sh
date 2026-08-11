# Shared blocked-path detection for the staging guard.
#
# Sourced by both pre-commit and commit-msg. They must agree exactly: if
# pre-commit blocks a file that commit-msg does not recognise, the override
# requirement silently stops applying — a guard that disagrees with itself is
# worse than none.

BLOCKED_PREFIXES=(
    "backups/"
    "deploy/"
    "outputs/"          # research output; force-added deliberately per CLAUDE.md
)

BLOCKED_PATTERNS=(
    '\.env$'
    '\.env\..*$'
    '\.dump$'
    '\.sql$'
    '\.sql\.gz$'
    '\.pem$'
    '\.key$'
    '\.p12$'
    '(^|/)skills-lock\.json$'
    '(^|/)id_(rsa|ed25519)$'
    '[ ][0-9]+\.(py|json|md|yml|csv)$'   # "foo 2.py" sync duplicates
)

# Print each staged path that lands in a protected location.
#
# Covers ADDs and RENAME DESTINATIONS. A rename was the hole: `--diff-filter=A`
# alone let a tracked file be moved to deploy/profile.env and committed with no
# override, because git records that as R rather than A.
#
# Plain modifications (M) of an already-tracked protected file stay allowed —
# once something legitimately lives in the repo, editing it must not demand the
# override every time, or the override becomes reflexive and stops meaning
# anything.
blocked_staged_paths() {
    local line status path
    git diff --cached --name-status --diff-filter=AR 2>/dev/null | while IFS= read -r line; do
        [ -z "$line" ] && continue
        status=$(printf '%s' "$line" | cut -f1)
        case "$status" in
            R*) path=$(printf '%s' "$line" | cut -f3) ;;   # rename destination
            *)  path=$(printf '%s' "$line" | cut -f2) ;;
        esac
        [ -z "$path" ] && continue

        local matched=""
        for prefix in "${BLOCKED_PREFIXES[@]}"; do
            case "$path" in
                "$prefix"*) matched="local-only path: $prefix"; break ;;
            esac
        done
        if [ -z "$matched" ]; then
            for pattern in "${BLOCKED_PATTERNS[@]}"; do
                if printf '%s' "$path" | grep -Eq "$pattern"; then
                    matched="blocked pattern: $pattern"
                    break
                fi
            done
        fi
        if [ -n "$matched" ]; then
            printf '%s  [%s]\n' "$path" "$matched"
        fi
    done
    # Always succeed. Under `set -e` in the calling hook, a non-zero status from
    # the final loop iteration — which happens whenever the last staged file is
    # clean — would abort the hook and reject a perfectly good commit with no
    # message at all.
    return 0
}
