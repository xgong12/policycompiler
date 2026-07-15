#!/usr/bin/env python3
"""
schemas.py -- the rule graph: block or allow, nothing between.

    rule    = {target_action, deny_when}
    body    = entry list, read as a conjunction; an empty list always fires
    entry   = {field, operator, value}   reads state
            | {not_completed: name}      reads completed_actions
    rules   = OR
    decision = DENY if any rule fires, ALLOW otherwise

This rule graph is a guardrail on an agent's tool call, and a guardrail answers one question: may this
call run now? So there are two outcomes, not three. A policy that would once have routed a
case to a human -- an over-limit charge, a stale report, a duplicate -- still blocks the
automated call, which is a DENY. Review is not a third verdict; it is what a person does
after the call is blocked, and that is outside the graph. Every rule therefore carries the
single head deny_when.

Anything outside the two entry shapes is rejected: no nesting, no `any`, no `unit`, no
`requires`, no `on_violation`. A rule that carries both heads, or neither, is rejected.
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPERATORS = (">", ">=", "<", "<=", "==", "!=")

Operator = Literal[">", ">=", "<", "<=", "==", "!="]
Decision = Literal["ALLOW", "DENY"]


class FieldEntry(BaseModel):
    """Reads one value out of state. A field the caller never sent does not hold."""
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: Operator
    value: Union[bool, int, float, str]


class NotCompletedEntry(BaseModel):
    """Reads the trail of tool calls. A name the caller never completed does hold."""
    model_config = ConfigDict(extra="forbid")

    not_completed: str


Entry = Annotated[Union[NotCompletedEntry, FieldEntry], Field(union_mode="left_to_right")]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_action: str
    deny_when: list[Entry]

    @property
    def verdict(self) -> str:
        return "DENY"

    @property
    def body(self) -> list:
        return self.deny_when


class RuleGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    rules: list[Rule]


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class RuntimeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed_actions: list[str] = Field(default_factory=list)
    state: dict = Field(default_factory=dict)
    proposed_action: ProposedAction
    expected_decision: Decision
