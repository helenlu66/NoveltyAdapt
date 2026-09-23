propose_operator_prompt = """You are a robot arm with a gripper that can manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, decide whether a missing operator that is EXECUTABLE in the current state could be proposed. Output `no operator` if there are existing operator(s) that should be executed first. Otherwise, propose 1 intuitive, non-existing operator involving the novel object(s) that is EXECUTABLE in the current state (not in the future) that would help make progress towards the goal. Output the proposed operator and the ground parameter objects by imitating the style of the existing operators in the following format. The parameters should include the gripper. Avoid quantifiers and conditional effects:
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
)
ground objects: object1, object2, object3 ...objectN
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
{current_state}
Goal state:
{goal_state}
Novel object(s) of interest: 
{novel_objects}
Specifically, the following atoms are true for the novel object(s):
{true_atoms_novel_obj}
The following atoms are false for the novel object(s):
{false_atoms_novel_obj}
Other relevant objects: 
{relevant_objects}
Object types: 
{object_types}
Existing operators with parameters:
{existing_operators}
```
Answer: Let's think step by step.
"""

propose_operator_prompt2 = """You are a robot arm with a gripper that can manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, propose 1 intuitive, non-existing operator involving the novel object(s) that is EXECUTABLE in the current state (not in the future) that would help make progress towards the goal. Output `no operator` if no immediate common sense operator need to be proposed i.e. there are existing operators applicable. Otherwise, output the proposed operator and the ground parameter objects by imitating the style of the existing operators in the following format. The parameters should include the gripper. Avoid quantifiers and conditional effects. The proposed operator should be a short horizon action. Avoid proposing a long horizon operator that requires manipulating multiple objects in a sequence:
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
)
ground objects: object1, object2, object3 ...objectN
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
{current_state}
Goal state:
{goal_state}
Novel object(s) of interest: 
{novel_objects}
Specifically, the following atoms are true for the novel object(s):
{true_atoms_novel_obj}
The following atoms are false for the novel object(s):
{false_atoms_novel_obj}
Other relevant objects: 
{relevant_objects}
Object types: 
{object_types}
Existing operators with parameters:
{existing_operators}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the precondition for the new operator
define_precondition_prompt = """You are a robot capable of understanding the Planning Domain Definition Language (PDDL). Given an operator's name, its parameter objects, the current states of parameter objects, and an image of the current state, fill in the preconditions of `The Operator` by selecting a relevant subset of atoms in the `Current state` section. The preconditions must ALREADY be satisfied in the current state. The preconditions MUST ONLY involve objects in `The Operator`'s parameters. Output the operator in the following format:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate1 ?param1 ?param2...) (predicate2 ?param1 ?param2...)...)
)
```
Problem:
```
Current state: 
{full_param_obj_atoms}
Example operators with parameters and preconditions:
{example_operators}
The Operator:
{proposed_operator}
```
Answer: Let's think step by step.
"""

# prompt asking the LLM to define the effect for the new operator
define_effect_prompt = """You are a robot arm with a gripper that uses the gripper to grasp and manipulate tabletop objects. Given a PDDL operator's name, its parameter objects, the preconditions that are satisfied in the current state, define the effects after applying the operator in the current state. The first step of manipulating an object is to make contact with it with your gripper. An object should exclusively occupy your gripper during manipulation. Therefore, if the object to manipulate is not occupying your gripper yet, you should prioritize making contact with it with your gripper. Your gripper should not be free while it is occupied. The effects MUST ONLY involve objects in The operator's parameters. Avoid quantifiers and conditional effects. Do not invent new predicates. Certain preconditions should be flipped in the effects. Think about the order in which the effects are expected to be acheived. Sort the effects in the order they are expected to be achieved. Fill the effects of The operator in their sorted order in the following format. MAKE SURE THERE ARE NO CONFLICTING EFFECTS:
```
(:action operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...)
    :precondition (and (predicate ?param1 ?param2...) (predicate ?param1 ?param2...)...)
    :effect (and (predicate ?param1 ?param2...) (predicate ?param1 ?param2...)...)
)
```
Problem:
```
Current state: 
{full_param_obj_atoms}
Example operators with parameters, preconditions, and effects:
{example_operators}
The operator:
{proposed_operator_with_precondition}
```
Answer: Let's think step by step.
"""

baseline_prompt = """You are a robot arm with a gripper that can manipulate tabletop objects. You are capable of understanding the Planning Domain Definition Language (PDDL). Given the current state, a set of objects deemed relevant to the task at hand, novel object(s) of interest, a set of existing operators, and a goal state, propose operator(s) that would help the achieve the goal. Output `no operator` if no new operators are necessary. Output the proposed operator and the ground parameter objects by imitating the style of the existing operators in the following format. The parameters should include the gripper. Avoid quantifiers and conditional effects:\n'
```
(:action proposed_non_existing_operator_name
    :parameters (?param1 - param1-type ?param2 - param2-type ...?paramN - paramN-type)
)
ground objects: object1, object2, object3 ...objectN
```
Problem:
```
Current state (unmentioned atoms are assumed false): 
 'attached(lid1,coffee-pod-holder1), can-flip-up(lid1), '
 'directly-on-table(drawer1,table1), directly-on-table(mug1,table1), '
 'free(gripper1), in(coffee-pod1,drawer1), open(mug1), '
 'small-enough-for-gripper-to-pick-up(coffee-pod-holder1,gripper1), '
 'small-enough-for-gripper-to-pick-up(coffee-pod1,gripper1), '
 'small-enough-for-gripper-to-pick-up(mug1,gripper1), '
 'under(mug1,coffee-pod-holder1), upright(mug1)\n'
 'Goal state:\n'
 'in(coffee-pod1,coffee-pod-holder1), under(mug1,coffee-pod-holder1)\n'
 'Novel object(s) of interest: \n'
 'drawer1\n'
 'Specifically, the following atoms are true for the novel object(s):\n'
 'in(coffee-pod1, drawer1), directly-on-table(drawer1, table1)\n'
 'The following atoms are false for the novel object(s):\n'
 'not(in(mug1, drawer1)), not(open(drawer1)), not(in(lid1, drawer1)), '
 'not(exclusively-occupying-gripper(drawer1, gripper1)), '
 'not(in(coffee-pod-holder1, drawer1)), '
 'not(small-enough-for-gripper-to-pick-up(drawer1, gripper1)), not(in(drawer1, '
 'mug1)), not(in(drawer1, coffee-pod-holder1))\n'
 'Other relevant objects: \n'
 'coffee-pod-holder1, drawer1, gripper1, lid1, table1, mug1, coffee-pod1\n'
 'Object types: \n'
 '(:types\n'
 '        gripper table tabletop-object - object\n'
 '        coffee-pod coffee-machine-lid container - tabletop-object\n'
 '        mug coffee-pod-holder drawer - container\n'
 '    \n'
 'Existing operators with parameters:\n'
 '(:action pick-up-from-tabletop\n'
 '\t:parameters (?gripper - gripper ?table - table ?tabletop-object - '
 'tabletop-object)\n'
 '\t:precondition (and (directly-on-table ?tabletop-object ?table) '
 '(small-enough-for-gripper-to-pick-up ?tabletop-object ?gripper) (free '
 '?gripper))\n'
 '\t:effect (and (exclusively-occupying-gripper ?tabletop-object ?gripper) '
 '(not (free ?gripper)) (not (directly-on-table ?tabletop-object ?table)))\n'
 ')\n'
 '(:action open-coffee-pod-holder\n'
 '\t:parameters (?gripper - gripper ?holder - coffee-pod-holder ?lid - '
 'coffee-machine-lid)\n'
 '\t:precondition (and (not (open ?holder)) (can-flip-up ?lid) (attached ?lid '
 '?holder) (free ?gripper))\n'
 '\t:effect (and (exclusively-occupying-gripper ?lid ?gripper) (not (free '
 '?gripper)) (open ?holder) (can-flip-down ?lid))\n'
 ')\n'
 '(:action close-coffee-pod-holder\n'
 '\t:parameters (?gripper - gripper ?holder - coffee-pod-holder ?lid - '
 'coffee-machine-lid)\n'
 '\t:precondition (and (open ?holder) (can-flip-down ?lid) (attached ?lid '
 '?holder) (free ?gripper))\n'
 '\t:effect (and (exclusively-occupying-gripper ?lid ?gripper) (not (free '
 '?gripper)) (not (open ?holder)))\n'
 ')\n'
 '(:action free-gripper-from-object-and-return-to-neutral\n'
 '\t:parameters (?gripper - gripper ?tabletop-object - tabletop-object)\n'
 '\t:precondition (and (not (small-enough-for-gripper-to-pick-up '
 '?tabletop-object ?gripper)) (exclusively-occupying-gripper ?tabletop-object '
 '?gripper) (not (free ?gripper)))\n'
 '\t:effect (and (not (exclusively-occupying-gripper ?tabletop-object '
 '?gripper)) (free ?gripper))\n'
 ')\n'
 '(:action place-mug-under-holder-from-gripper\n'
 '\t:parameters (?gripper - gripper ?holder - coffee-pod-holder ?mug - mug)\n'
 '\t:precondition (and (exclusively-occupying-gripper ?mug ?gripper) (not '
 '(free ?gripper)))\n'
 '\t:effect (and (under ?mug ?holder) (upright ?mug) (not '
 '(exclusively-occupying-gripper ?mug ?gripper)) (free ?gripper))\n'
 ')\n'
 '```\n'
 "Answer: Let's think step by step.\n"""


    