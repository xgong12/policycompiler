# PolicyCompiler:`deny_when` 的求值逻辑怎么读

> 一份把「缺失输入 → ALLOW 还是 DENY」讲清楚的说明。核心在于:`deny_when` 里有**两种**条件,它们对「缺失」的含义天然相反——一个是「陈述事实」,一个是「否定式的前置要求」。

---

## 引擎的两个关键函数(`rule_engine.py`)

```python
def entry_holds(entry, state, completed_actions):
    if isinstance(entry, NotCompletedEntry):          # 前置步骤(否定语义)
        name = entry.not_completed
        return name not in completed_actions and not state.get(name)
    if entry.field not in state:                      # 字段事实
        return False
    return COMPARISONS[entry.operator](state[entry.field], entry.value)

def rule_fires(rule, state, completed_actions):
    return all(entry_holds(e, state, completed_actions) for e in rule.body)   # AND

# check_action(...):  decision = "DENY" if 有规则 fire else "ALLOW"
```

一条规则(rule)**当且仅当它 body 里每个条件都成立**时才 fire(`all(...)` = AND);任何一条规则 fire → 该动作 **DENY**;没有任何规则 fire → **ALLOW**。

---

## 两种条件,对「缺失」的处理相反

### ① field entry —— 描述一个**事实**
形如 `{field, operator, value}`,例如 `expense_category == hotel`。

- **读法**:字段不在 `state` 里 → 返回 `False` → `all(False, …) = False` → 规则不 fire → **ALLOW**。
- **直觉**:规则对它没被告知的事实**保持沉默**,不会误拦。
- 这叫 **fail-open**(缺信息 → 放行)。

### ② prerequisite entry —— 一个**否定式的前置要求**
形如 `{not_completed: "manager_approval"}`。注意它是**否定语义**:整条的意思是「manager_approval **尚未完成**」。

- **读法**:`manager_approval` 不在 `completed_actions` 里 → `name not in completed_actions` 为 `True` → **条件成立(缺失 = 命中)** → 贡献 fire → **DENY**。
- **直觉**:必须先做的步骤没做,就**拦下来**。
- 这叫 **fail-closed**(缺步骤 → 拦截)。

> **关键点**:`not_completed` 是「否定条件」,前置步骤**缺失时它返回 `True`**,而不是 `False`。这就是为什么「缺失前置 → DENY」。它和 field entry(缺失 → `False`)方向正好相反。

---

## 一个具体例子

策略「酒店报销需经理审批」编译成的 structured rule:

```json
{
  "target_action": "issue_reimbursement",
  "deny_when": [
    {"field": "expense_category", "operator": "==", "value": "hotel"},
    {"not_completed": "manager_approval"}
  ]
}
```

**情形 A —— 经理还没批**
`state = {expense_category: "hotel"}`,`completed_actions = []`

- entry1(field):`expense_category == hotel` → **True**
- entry2(not_completed):`manager_approval` 不在已完成列表 → **True**(即「它缺失」)
- `all(True, True) = True` → 规则 fire → **DENY** ✅

**情形 B —— 经理已批**
`completed_actions = ["manager_approval"]`

- entry2 变 **False**(步骤已完成)→ `all(...) = False` → 不 fire → **ALLOW** ✅

**情形 C —— 报销的根本不是酒店(比如缺 `expense_category` 或值是 meals)**

- entry1(field)不成立 → `all(False, …) = False` → 不 fire → **ALLOW**(这条酒店规则对它保持沉默)

---

## 一句话总结

- **事实类条件(field):缺信息 → 沉默 → 放行**(fail-open)
- **步骤类条件(not_completed):缺步骤 → 命中 → 拦截**(fail-closed)

README 里的 **"fields fail open, prerequisites fail closed"** 和代码逐行吻合。全部区别就在 `not_completed` 这个**否定语义**上——它「缺失即为真」。这个非对称是刻意的安全设计:描述「事实」的条件缺信息就沉默(不误拦),而「必须先做的步骤」缺失就拦截(不放水)。
