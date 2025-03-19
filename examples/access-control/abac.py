"""
Attribute Based Access Control (ABAC) Demo with Formal Verification

1. A basic ABAC model implementation
2. Formal verification of ABAC policies using Z3
3. Property specification and checking for ABAC systems
"""

from z3 import *
import itertools
from typing import Dict, List, Set, Tuple, Optional, Union


class AttributeBasedAccessControl:
    """
    Implementation of an Attribute Based Access Control system that can be
    formally verified using Z3.
    """
    
    def __init__(self):
        # Core ABAC components
        self.subjects = set()  # Users or entities requesting access
        self.resources = set()  # Objects being accessed
        self.actions = set()  # Operations that can be performed on resources
        self.environments = set()  # Contextual information
        
        # Attributes for each component
        self.subject_attributes = {}  # user_id → {attr_name: attr_value}
        self.resource_attributes = {}  # resource_id → {attr_name: attr_value}
        self.action_attributes = {}  # action_id → {attr_name: attr_value}
        self.environment_attributes = {}  # env_id → {attr_name: attr_value}
        
        # Policies defined as conditions on attributes
        self.policies = []
    
    def add_subject(self, subject_id: str, attributes: Dict[str, str]):
        """Add a subject with attributes to the system."""
        self.subjects.add(subject_id)
        self.subject_attributes[subject_id] = attributes
    
    def add_resource(self, resource_id: str, attributes: Dict[str, str]):
        """Add a resource with attributes to the system."""
        self.resources.add(resource_id)
        self.resource_attributes[resource_id] = attributes
    
    def add_action(self, action_id: str, attributes: Dict[str, str]):
        """Add an action with attributes to the system."""
        self.actions.add(action_id)
        self.action_attributes[action_id] = attributes
    
    def add_environment(self, env_id: str, attributes: Dict[str, str]):
        """Add an environment context with attributes to the system."""
        self.environments.add(env_id)
        self.environment_attributes[env_id] = attributes
    
    def add_policy(self, policy_function):
        """
        Add a policy to the system.
        The policy_function should take (subject, resource, action, environment) and return boolean.
        """
        self.policies.append(policy_function)
    
    def check_access(self, subject_id: str, resource_id: str, action_id: str, 
                     environment_id: str) -> bool:
        """
        Check if a subject can perform an action on a resource in a given environment.
        Returns True if access is granted, False otherwise.
        """
        if (subject_id not in self.subjects or 
            resource_id not in self.resources or 
            action_id not in self.actions or
            environment_id not in self.environments):
            return False
        
        subject_attrs = self.subject_attributes[subject_id]
        resource_attrs = self.resource_attributes[resource_id]
        action_attrs = self.action_attributes[action_id]
        env_attrs = self.environment_attributes[environment_id]
        
        # For access to be granted, all policies must be satisfied
        for policy in self.policies:
            if not policy(subject_attrs, resource_attrs, action_attrs, env_attrs):
                return False
        
        return True


class ABACVerifier:
    """Formal verification of ABAC policies using Z3."""
    
    def __init__(self, abac_system: AttributeBasedAccessControl):
        self.abac = abac_system
        self.solver = Solver()
        
        # Z3 sorts for our domain
        self.SubjectSort = DeclareSort('Subject')
        self.ResourceSort = DeclareSort('Resource')
        self.ActionSort = DeclareSort('Action')
        self.EnvSort = DeclareSort('Environment')
        
        # Constants for each entity in our system
        self.subject_constants = {}
        self.resource_constants = {}
        self.action_constants = {}
        self.env_constants = {}
        
        # Attribute functions for Z3
        self.subject_attr_funcs = {}
        self.resource_attr_funcs = {}
        self.action_attr_funcs = {}
        self.env_attr_funcs = {}
        
        # Initialize the model
        self._initialize_constants()
        self._initialize_attribute_functions()
    
    def _initialize_constants(self):
        """Create Z3 constants for all entities in the ABAC system."""
        for subject_id in self.abac.subjects:
            self.subject_constants[subject_id] = Const(f"subject_{subject_id}", self.SubjectSort)
            
        for resource_id in self.abac.resources:
            self.resource_constants[resource_id] = Const(f"resource_{resource_id}", self.ResourceSort)
            
        for action_id in self.abac.actions:
            self.action_constants[action_id] = Const(f"action_{action_id}", self.ActionSort)
            
        for env_id in self.abac.environments:
            self.env_constants[env_id] = Const(f"env_{env_id}", self.EnvSort)
    
    def _initialize_attribute_functions(self):
        """Initialize Z3 functions for attributes."""
        # Collect all attribute names
        subject_attrs = set()
        for attrs in self.abac.subject_attributes.values():
            subject_attrs.update(attrs.keys())
            
        resource_attrs = set()
        for attrs in self.abac.resource_attributes.values():
            resource_attrs.update(attrs.keys())
            
        action_attrs = set()
        for attrs in self.abac.action_attributes.values():
            action_attrs.update(attrs.keys())
            
        env_attrs = set()
        for attrs in self.abac.environment_attributes.values():
            env_attrs.update(attrs.keys())
        
        # Create functions for each attribute
        for attr in subject_attrs:
            self.subject_attr_funcs[attr] = Function(f"subject_{attr}", self.SubjectSort, StringSort())
            
        for attr in resource_attrs:
            self.resource_attr_funcs[attr] = Function(f"resource_{attr}", self.ResourceSort, StringSort())
            
        for attr in action_attrs:
            self.action_attr_funcs[attr] = Function(f"action_{attr}", self.ActionSort, StringSort())
            
        for attr in env_attrs:
            self.env_attr_funcs[attr] = Function(f"env_{attr}", self.EnvSort, StringSort())
    
    def define_attribute_constraints(self):
        """Define constraints based on attribute values in the system."""
        constraints = []
        
        # Add constraints for subject attributes
        for subject_id, attrs in self.abac.subject_attributes.items():
            subject = self.subject_constants[subject_id]
            for attr_name, attr_value in attrs.items():
                if attr_name in self.subject_attr_funcs:
                    constraints.append(
                        self.subject_attr_funcs[attr_name](subject) == StringVal(attr_value)
                    )
        
        # Add constraints for resource attributes
        for resource_id, attrs in self.abac.resource_attributes.items():
            resource = self.resource_constants[resource_id]
            for attr_name, attr_value in attrs.items():
                if attr_name in self.resource_attr_funcs:
                    constraints.append(
                        self.resource_attr_funcs[attr_name](resource) == StringVal(attr_value)
                    )
        
        # Add constraints for action attributes
        for action_id, attrs in self.abac.action_attributes.items():
            action = self.action_constants[action_id]
            for attr_name, attr_value in attrs.items():
                if attr_name in self.action_attr_funcs:
                    constraints.append(
                        self.action_attr_funcs[attr_name](action) == StringVal(attr_value)
                    )
        
        # Add constraints for environment attributes
        for env_id, attrs in self.abac.environment_attributes.items():
            env = self.env_constants[env_id]
            for attr_name, attr_value in attrs.items():
                if attr_name in self.env_attr_funcs:
                    constraints.append(
                        self.env_attr_funcs[attr_name](env) == StringVal(attr_value)
                    )
        
        return constraints
    
    def access_function(self):
        """Create a Z3 function representing the access control decision."""
        return Function('access', self.SubjectSort, self.ResourceSort, 
                        self.ActionSort, self.EnvSort, BoolSort())
    
    def policy_to_z3(self, policy):
        """
        Convert a policy function to Z3 constraints.
        This is a simplified implementation and will need customization for actual policies.
        """
        access = self.access_function()
        constraints = []
        
        # Iterate over all possible combinations of entities
        for s_id, r_id, a_id, e_id in itertools.product(
            self.abac.subjects, self.abac.resources, 
            self.abac.actions, self.abac.environments):
            
            s = self.subject_constants[s_id]
            r = self.resource_constants[r_id]
            a = self.action_constants[a_id]
            e = self.env_constants[e_id]
            
            # Construct policy in Z3 based on attribute values
            s_attrs = self.abac.subject_attributes[s_id]
            r_attrs = self.abac.resource_attributes[r_id]
            a_attrs = self.abac.action_attributes[a_id]
            e_attrs = self.abac.environment_attributes[e_id]
            
            # Check if access should be granted based on the policy
            if policy(s_attrs, r_attrs, a_attrs, e_attrs):
                constraints.append(access(s, r, a, e) == True)
            else:
                constraints.append(access(s, r, a, e) == False)
        
        return constraints
    
    def verify_property(self, property_formula):
        """
        Verify if a property holds in the system.
        
        Args:
            property_formula: A Z3 formula expressing a property to verify
            
        Returns:
            (result, model): result is sat/unsat/unknown, model is a counterexample if sat
        """
        self.solver.push()
        
        # Add attribute constraints
        for constraint in self.define_attribute_constraints():
            self.solver.add(constraint)
        
        # Add policy constraints
        for policy in self.abac.policies:
            for constraint in self.policy_to_z3(policy):
                self.solver.add(constraint)
        
        # Add the negation of the property to find a counterexample
        self.solver.add(Not(property_formula))
        
        result = self.solver.check()
        model = None
        if result == sat:
            model = self.solver.model()
        
        self.solver.pop()
        return result, model
    
    def verify_separation_of_duty(self, sensitive_resource, role1, role2):
        """
        Verify that users with role1 cannot access sensitive_resource if they also have role2.
        This is a common separation of duty constraint.
        """
        access = self.access_function()
        
        subject = Const("subject_var", self.SubjectSort)
        action = Const("action_var", self.ActionSort)
        env = Const("env_var", self.EnvSort)
        
        property_formula = ForAll([subject, action, env], 
            Implies(
                And(
                    self.subject_attr_funcs['role'](subject) == StringVal(role1),
                    self.subject_attr_funcs['role'](subject) == StringVal(role2)
                ),
                Not(access(subject, self.resource_constants[sensitive_resource], 
                          action, env))
            )
        )
        
        return self.verify_property(property_formula)
    
    def verify_principle_of_least_privilege(self, role, necessary_actions, all_actions):
        """
        Verify that users with a specific role can only perform necessary actions
        and not other actions on resources.
        """
        access = self.access_function()
        
        subject = Const("subject_var", self.SubjectSort)
        resource = Const("resource_var", self.ResourceSort)
        env = Const("env_var", self.EnvSort)
        
        # Create constraints for each action
        constraints = []
        for action_id in all_actions:
            action = self.action_constants[action_id]
            
            if action_id in necessary_actions:
                # Should be able to access
                formula = ForAll([subject, resource, env],
                    Implies(
                        self.subject_attr_funcs['role'](subject) == StringVal(role),
                        access(subject, resource, action, env)
                    )
                )
            else:
                # Should NOT be able to access
                formula = ForAll([subject, resource, env],
                    Implies(
                        self.subject_attr_funcs['role'](subject) == StringVal(role),
                        Not(access(subject, resource, action, env))
                    )
                )
            
            constraints.append(formula)
        
        # All constraints must hold
        property_formula = And(constraints)
        return self.verify_property(property_formula)


# Example usage
def example_usage():
    # Create ABAC system
    abac = AttributeBasedAccessControl()
    
    # Add subjects (users)
    abac.add_subject("alice", {
        "role": "manager",
        "department": "finance",
        "clearance": "high"
    })
    
    abac.add_subject("bob", {
        "role": "employee",
        "department": "engineering",
        "clearance": "medium"
    })
    
    abac.add_subject("charlie", {
        "role": "contractor",
        "department": "engineering",
        "clearance": "low"
    })
    
    # Add resources
    abac.add_resource("financial_report", {
        "type": "document",
        "sensitivity": "high",
        "department": "finance"
    })
    
    abac.add_resource("code_repo", {
        "type": "repository",
        "sensitivity": "medium",
        "department": "engineering"
    })
    
    abac.add_resource("company_wiki", {
        "type": "knowledge_base",
        "sensitivity": "low",
        "department": "all"
    })
    
    # Add actions
    abac.add_action("read", {"impact": "low"})
    abac.add_action("write", {"impact": "high"})
    abac.add_action("execute", {"impact": "medium"})
    
    # Add environment contexts
    abac.add_environment("office_hours", {
        "time": "work_hours",
        "location": "office",
        "connection": "secure"
    })
    
    abac.add_environment("remote", {
        "time": "any",
        "location": "remote",
        "connection": "vpn"
    })
    
    # Define policies
    
    # Policy 1: Subject can access resource if they are in the same department
    def department_policy(subj_attrs, res_attrs, act_attrs, env_attrs):
        return (res_attrs["department"] == subj_attrs["department"] or 
                res_attrs["department"] == "all")
    
    # Policy 2: Subject's clearance must be at least as high as resource sensitivity
    def clearance_policy(subj_attrs, res_attrs, act_attrs, env_attrs):
        clearance_levels = {"low": 1, "medium": 2, "high": 3}
        return clearance_levels[subj_attrs["clearance"]] >= clearance_levels[res_attrs["sensitivity"]]
    
    # Policy 3: High impact actions require high clearance
    def impact_policy(subj_attrs, res_attrs, act_attrs, env_attrs):
        if act_attrs["impact"] == "high":
            return subj_attrs["clearance"] == "high"
        return True
    
    # Policy 4: Remote access requires VPN connection
    def remote_policy(subj_attrs, res_attrs, act_attrs, env_attrs):
        if env_attrs["location"] == "remote":
            return env_attrs["connection"] == "vpn"
        return True
    
    # Add policies to the system
    abac.add_policy(department_policy)
    abac.add_policy(clearance_policy)
    abac.add_policy(impact_policy)
    abac.add_policy(remote_policy)
    
    # Test access
    print("Testing access control decisions:")
    print("-" * 40)
    
    # Alice (manager, finance, high) accessing financial_report
    result = abac.check_access("alice", "financial_report", "write", "office_hours")
    print(f"Alice writing financial report during office hours: {result}")
    
    # Bob (employee, engineering, medium) accessing code_repo
    result = abac.check_access("bob", "code_repo", "read", "remote")
    print(f"Bob reading code repo remotely: {result}")
    
    # Charlie (contractor, engineering, low) accessing code_repo to write
    result = abac.check_access("charlie", "code_repo", "write", "office_hours")
    print(f"Charlie writing to code repo during office hours: {result}")
    
    # Bob accessing financial_report
    result = abac.check_access("bob", "financial_report", "read", "office_hours")
    print(f"Bob reading financial report during office hours: {result}")
    
    print("\nFormal Verification Example:")
    print("-" * 40)
    
    # Create a verifier
    verifier = ABACVerifier(abac)
    
    # Example property: A subject with low clearance cannot access high sensitivity resources
    access = verifier.access_function()
    
    # Create variables of the proper sort types for the formula
    subject = Const("subject_var", verifier.SubjectSort)
    resource = Const("resource_var", verifier.ResourceSort)
    action = Const("action_var", verifier.ActionSort)
    env = Const("env_var", verifier.EnvSort)
    
    property_formula = ForAll(
        [subject, resource, action, env],
        Implies(
            And(
                verifier.subject_attr_funcs["clearance"](subject) == StringVal("low"),
                verifier.resource_attr_funcs["sensitivity"](resource) == StringVal("high")
            ),
            Not(access(subject, resource, action, env))
        )
    )
    
    result, model = verifier.verify_property(property_formula)
    print(f"Property verification result: {result}")
    if result == sat:
        print("Property violated. Counterexample found:")
        print(model)
    else:
        print("Property holds. No violations possible.")


if __name__ == "__main__":
    example_usage()


