"""Response Composer: template + slot-fill + rank. Never a generator —
every branch below picks/formats a fixed template with retrieved data.

Returns a tuple: (reply_text, conv_entry_id | None)
The conv_entry_id is used by ChatSession to prevent repeating the same
chitchat entry consecutively.
"""
import random

_CLARIFY_TEMPLATES = [
    "I'm not sure I understood that — could you rephrase?",
    "Sorry, I didn't quite catch that. Could you say it differently?",
    "Hmm, I'm not sure what you mean. Could you try asking differently?",
    "I'm a bit confused by that one. Can you rephrase?",
]


def compose(route_result: dict, state, recent_conv_ids=None) -> tuple[str, str | None]:
    """Compose a reply and return (reply_text, conv_entry_id).
    conv_entry_id is None for all non-conversation subsystems.
    recent_conv_ids: deque or set of recently used entry IDs to avoid repeating."""
    subsystem = route_result["subsystem"]
    payload = route_result["payload"]
    is_child = (state.persona == "child")
    excluded = set(recent_conv_ids) if recent_conv_ids else set()

    if subsystem == "persona_switch":
        return payload["reply"], None

    if subsystem == "reset":
        if is_child:
            return "Okay! Starting fresh. What would you like to talk about now? 🌟", None
        return "Okay, starting fresh — what would you like help with?", None

    if subsystem == "smalltalk":
        if is_child and payload.get("intent") == "casual":
            return "That's okay! We can talk about whatever you want, or ask for a joke! 🎈", None
        return payload["reply"], None

    if subsystem == "math":
        if is_child:
            return f"The math result is: {payload['result']}! 🧮✨", None
        return f"Calculated (via C++ High-Speed Math Engine): {payload['result']}", None

    if subsystem == "flow":
        return payload["text"], None

    if subsystem == "conversation":
        results = payload["results"]
        top_score = results[0]["score"]

        # Wide window: 0.15 gap so tag-expanded entries (with slight discount) also qualify
        candidates = [r for r in results if (top_score - r["score"]) <= 0.15]

        # Anti-repeat: exclude all recently seen IDs; fall back to full pool if all excluded
        fresh = [c for c in candidates if c["id"] not in excluded]
        pool = fresh if fresh else candidates

        chosen = random.choice(pool)
        conv_id = chosen["id"]

        if is_child and "child_answer" in chosen:
            return chosen["child_answer"], conv_id
        return chosen["answer"], conv_id

    if subsystem == "generative":
        return payload["text"], None

    if subsystem == "kb":
        results = payload["results"]
        top = results[0]
        if len(results) == 1 or top["score"] >= 0.6:
            if is_child and "child_answer" in top:
                return top["child_answer"], None
            return top["answer"], None
        # Ambiguous: show disambiguation list
        if is_child:
            lines = ["I found a few things you might be asking about:"]
            for r in results[:3]:
                lines.append(f"- {r['question']}: {r.get('child_answer', r['answer'])}")
        else:
            lines = ["I found a few things that might match:"]
            for r in results[:3]:
                lines.append(f"- {r['question']}: {r['answer']}")
        return "\n".join(lines), None

    if subsystem == "web":
        if payload.get("error"):
            if is_child:
                return "I couldn't find anything on that. Maybe ask a parent to look it up? 🧐", None
            return "I couldn't find anything on that.", None
        summary = payload.get("summary")
        if summary:
            # High-speed C++ TextSanitizer pass
            try:
                from . import minibrain_cpp as mb
                summary = mb.TextSanitizer.sanitize(summary)
            except Exception:
                pass

            # Enforce clean concise output cap (max 380 chars)
            if len(summary) > 380:
                cut = summary[:380]
                end = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
                if end > 120:
                    summary = cut[:end+1]
                else:
                    summary = cut + "..."
            sources = payload.get("sources", [])
            link = sources[0] if sources else None

            if is_child:
                reply = f"Here is what I found for you: {summary} ✨"
                if link:
                    reply += f"\nYou can read more here: {link}"
                return reply, None

            reply = summary
            if link:
                reply += f"\n\nSource: {link}"
            return reply, None
        results = payload.get("results", [])
        if not results:
            if is_child:
                return "I couldn't find anything on that. Maybe ask a parent to look it up? 🧐", None
            return "I couldn't find anything on that.", None
        if len(results) == 1:
            r = results[0]
            snippet = f"Here's what I found: {r['snippet']} — {r['url']}".strip(" —")
            return snippet, None
        lines = ["Here's what I found:"]
        for r in results[:3]:
            lines.append(f"- {r['title']}: {r['snippet']} ({r['url']})")
        return "\n".join(lines), None

    # Clarify / fallback
    if is_child:
        return "I'm not sure I got that. Could you say it in a simpler way? 🤔", None
    return random.choice(_CLARIFY_TEMPLATES), None