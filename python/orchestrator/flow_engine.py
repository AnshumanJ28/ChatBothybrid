"""Flow Engine: multi-step slot-filling state machine.

Handles structured tasks (e.g. 'return an item') that need more than
one turn to collect required slots. Still purely selection/state
tracking — it picks the next prompt from a template table keyed by
(flow, slot), never generates one.
"""

# Each flow declares required slots in order and a prompt template per slot.
FLOW_DEFINITIONS = {
    "return_item": {
        "trigger_keywords": {"return", "refund", "exchange"},
        "slots": ["order_id", "reason"],
        "prompts": {
            "order_id": "Sure, I can help with a return. What's your order ID?",
            "reason": "Got it. What's the reason for the return?",
        },
        "completion_template": (
            "Thanks — I've started a return for order {order_id} "
            "(reason: {reason}). You'll get a confirmation email shortly."
        ),
    },
    "book_demo": {
        "trigger_keywords": {"demo", "schedule", "book"},
        "slots": ["company_name", "preferred_date"],
        "prompts": {
            "company_name": "Happy to set up a demo. What's your company name?",
            "preferred_date": "What date works best for you?",
        },
        "completion_template": (
            "Great, I've requested a demo for {company_name} on {preferred_date}. "
            "Someone from our team will confirm shortly."
        ),
    },
}


def match_trigger(tokens: list[str]) -> str | None:
    """Returns a flow name if tokens match a flow's trigger keywords."""
    token_set = set(tokens)
    for flow_name, definition in FLOW_DEFINITIONS.items():
        if token_set & definition["trigger_keywords"]:
            return flow_name
    return None


class FlowInstance:
    """Tracks progress through one active flow for one session."""

    def __init__(self, flow_name: str):
        self.flow_name = flow_name
        self.definition = FLOW_DEFINITIONS[flow_name]
        self.slots: dict[str, str] = {}
        self.current_slot_index = 0

    @property
    def is_complete(self) -> bool:
        return self.current_slot_index >= len(self.definition["slots"])

    def next_prompt(self) -> str:
        slot = self.definition["slots"][self.current_slot_index]
        return self.definition["prompts"][slot]

    def fill_next_slot(self, raw_value: str) -> str:
        """Fills the current slot with the raw user text and advances.
        Returns the prompt for the next slot, or the completion message
        if the flow is now done."""
        slot = self.definition["slots"][self.current_slot_index]
        self.slots[slot] = raw_value.strip()
        self.current_slot_index += 1

        if self.is_complete:
            return self.definition["completion_template"].format(**self.slots)
        return self.next_prompt()
