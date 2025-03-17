"""
Role-Based Access Control (RBAC) Demo with Formal Verification

This module demonstrates:
1. A basic RBAC model implementation
2. Formal verification of RBAC policies using Z3
3. Property specification and checking for RBAC systems

The implementation allows for:
- Defining users, roles, permissions, and resources
- Creating role hierarchies
- Specifying and verifying security properties
- Checking for policy conflicts and violations
"""

from z3 import *
import itertools
from typing import Dict, List, Set, Tuple, Optional, Union


class RBACModel:
    """
    A Role-Based Access Control model with formal verification capabilities.
    """
    
    def __init__(self):
        # Core RBAC components
        self.users = set()
        self.roles = set()
        self.permissions = set()
        self.resources = set()
        
        # Assignments
        self.user_role_assignments = {}  # user -> set of roles
        self.role_permission_assignments = {}  # role -> set of permissions
        self.permission_resource_assignments = {}  # permission -> set of resources
        
        # Role hierarchy (role -> set of sub-roles)
        self.role_hierarchy = {}
        
        # Z3 solver for verification
        self.solver = Solver()
        
        # Z3 variables
        self.z3_users = {}
        self.z3_roles = {}
        self.z3_permissions = {}
        self.z3_resources = {}
        self.z3_user_role = {}
        self.z3_role_perm = {}
        self.z3_perm_resource = {}
        self.z3_role_hierarchy = {}
        
    def add_user(self, user: str) -> None:
        """Add a user to the RBAC model."""
        self.users.add(user)
        self.user_role_assignments[user] = set()
        
    def add_role(self, role: str) -> None:
        """Add a role to the RBAC model."""
        self.roles.add(role)
        self.role_permission_assignments[role] = set()
        self.role_hierarchy[role] = set()
        
    def add_permission(self, permission: str) -> None:
        """Add a permission to the RBAC model."""
        self.permissions.add(permission)
        self.permission_resource_assignments[permission] = set()
        
    def add_resource(self, resource: str) -> None:
        """Add a resource to the RBAC model."""
        self.resources.add(resource)
        
    def assign_user_to_role(self, user: str, role: str) -> None:
        """Assign a user to a role."""
        if user not in self.users:
            raise ValueError(f"User '{user}' does not exist")
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
            
        self.user_role_assignments[user].add(role)
        
    def assign_permission_to_role(self, permission: str, role: str) -> None:
        """Assign a permission to a role."""
        if permission not in self.permissions:
            raise ValueError(f"Permission '{permission}' does not exist")
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
            
        self.role_permission_assignments[role].add(permission)
        
    def assign_permission_to_resource(self, permission: str, resource: str) -> None:
        """Assign a permission to a resource."""
        if permission not in self.permissions:
            raise ValueError(f"Permission '{permission}' does not exist")
        if resource not in self.resources:
            raise ValueError(f"Resource '{resource}' does not exist")
            
        self.permission_resource_assignments[permission].add(resource)
        
    def add_role_hierarchy(self, senior_role: str, junior_role: str) -> None:
        """
        Add a role hierarchy relationship where senior_role inherits permissions from junior_role.
        """
        if senior_role not in self.roles:
            raise ValueError(f"Role '{senior_role}' does not exist")
        if junior_role not in self.roles:
            raise ValueError(f"Role '{junior_role}' does not exist")
            
        self.role_hierarchy[senior_role].add(junior_role)
        
    def get_all_permissions_for_role(self, role: str) -> Set[str]:
        """
        Get all permissions for a role, including those inherited from junior roles.
        """
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
            
        # Direct permissions
        permissions = set(self.role_permission_assignments[role])
        
        # Inherited permissions
        for junior_role in self.role_hierarchy[role]:
            permissions.update(self.get_all_permissions_for_role(junior_role))
            
        return permissions
        
    def get_all_roles_for_user(self, user: str) -> Set[str]:
        """
        Get all roles for a user, including those inherited through role hierarchy.
        """
        if user not in self.users:
            raise ValueError(f"User '{user}' does not exist")
            
        # Direct roles
        roles = set(self.user_role_assignments[user])
        
        # Add senior roles
        all_roles = set(roles)
        for role in roles:
            for senior_role in self.roles:
                if role in self.get_all_junior_roles(senior_role):
                    all_roles.add(senior_role)
                    
        return all_roles
    
    def get_all_junior_roles(self, role: str) -> Set[str]:
        """
        Get all junior roles for a given role.
        """
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
            
        junior_roles = set(self.role_hierarchy[role])
        for jr in list(junior_roles):
            junior_roles.update(self.get_all_junior_roles(jr))
            
        return junior_roles
        
    def check_user_permission(self, user: str, permission: str) -> bool:
        """
        Check if a user has a specific permission.
        """
        if user not in self.users:
            raise ValueError(f"User '{user}' does not exist")
        if permission not in self.permissions:
            raise ValueError(f"Permission '{permission}' does not exist")
            
        user_roles = self.get_all_roles_for_user(user)
        
        for role in user_roles:
            role_permissions = self.get_all_permissions_for_role(role)
            if permission in role_permissions:
                return True
                
        return False
        
    def check_user_access_to_resource(self, user: str, resource: str) -> bool:
        """
        Check if a user has access to a specific resource.
        """
        if user not in self.users:
            raise ValueError(f"User '{user}' does not exist")
        if resource not in self.resources:
            raise ValueError(f"Resource '{resource}' does not exist")
            
        user_roles = self.get_all_roles_for_user(user)
        
        for role in user_roles:
            role_permissions = self.get_all_permissions_for_role(role)
            for permission in role_permissions:
                if resource in self.permission_resource_assignments[permission]:
                    return True
                    
        return False
    
    # Formal verification methods
    
    def initialize_z3_variables(self):
        """
        Initialize Z3 variables for formal verification.
        """
        # Create boolean variables for each entity
        for user in self.users:
            self.z3_users[user] = Bool(f"user_{user}")
            
        for role in self.roles:
            self.z3_roles[role] = Bool(f"role_{role}")
            
        for permission in self.permissions:
            self.z3_permissions[permission] = Bool(f"permission_{permission}")
            
        for resource in self.resources:
            self.z3_resources[resource] = Bool(f"resource_{resource}")
            
        # Create variables for relationships
        for user in self.users:
            self.z3_user_role[user] = {}
            for role in self.roles:
                self.z3_user_role[user][role] = Bool(f"user_{user}_has_role_{role}")
                
        for role in self.roles:
            self.z3_role_perm[role] = {}
            for permission in self.permissions:
                self.z3_role_perm[role][permission] = Bool(f"role_{role}_has_permission_{permission}")
                
        for permission in self.permissions:
            self.z3_perm_resource[permission] = {}
            for resource in self.resources:
                self.z3_perm_resource[permission][resource] = Bool(f"permission_{permission}_on_resource_{resource}")
                
        for senior_role in self.roles:
            self.z3_role_hierarchy[senior_role] = {}
            for junior_role in self.roles:
                self.z3_role_hierarchy[senior_role][junior_role] = Bool(f"role_{senior_role}_inherits_{junior_role}")
    
    def encode_rbac_model(self):
        """
        Encode the RBAC model into Z3 constraints.
        """
        self.initialize_z3_variables()
        
        # Reset solver
        self.solver = Solver()
        
        # Encode user-role assignments
        for user in self.users:
            for role in self.roles:
                if role in self.user_role_assignments[user]:
                    self.solver.add(self.z3_user_role[user][role])
                else:
                    self.solver.add(Not(self.z3_user_role[user][role]))
                    
        # Encode role-permission assignments
        for role in self.roles:
            for permission in self.permissions:
                if permission in self.role_permission_assignments[role]:
                    self.solver.add(self.z3_role_perm[role][permission])
                else:
                    self.solver.add(Not(self.z3_role_perm[role][permission]))
                    
        # Encode permission-resource assignments
        for permission in self.permissions:
            for resource in self.resources:
                if resource in self.permission_resource_assignments[permission]:
                    self.solver.add(self.z3_perm_resource[permission][resource])
                else:
                    self.solver.add(Not(self.z3_perm_resource[permission][resource]))
                    
        # Encode role hierarchy
        for senior_role in self.roles:
            for junior_role in self.roles:
                if junior_role in self.role_hierarchy[senior_role]:
                    self.solver.add(self.z3_role_hierarchy[senior_role][junior_role])
                else:
                    self.solver.add(Not(self.z3_role_hierarchy[senior_role][junior_role]))
                    
        # Encode role hierarchy transitivity
        for r1 in self.roles:
            for r2 in self.roles:
                for r3 in self.roles:
                    # If r1 inherits r2 and r2 inherits r3, then r1 inherits r3
                    self.solver.add(
                        Implies(
                            And(
                                self.z3_role_hierarchy[r1][r2],
                                self.z3_role_hierarchy[r2][r3]
                            ),
                            self.z3_role_hierarchy[r1][r3]
                        )
                    )
                    
        # Encode permission inheritance through role hierarchy
        for senior_role in self.roles:
            for junior_role in self.roles:
                for permission in self.permissions:
                    # If senior_role inherits junior_role and junior_role has permission,
                    # then senior_role effectively has that permission
                    self.solver.add(
                        Implies(
                            And(
                                self.z3_role_hierarchy[senior_role][junior_role],
                                self.z3_role_perm[junior_role][permission]
                            ),
                            # This is an "effective" permission, not directly assigned
                            # We don't add it to the direct assignments
                            True
                        )
                    )
    
    def verify_property(self, property_formula):
        """
        Verify a property against the RBAC model.
        
        Args:
            property_formula: A Z3 formula representing the property to verify
            
        Returns:
            (bool, Optional[dict]): A tuple containing:
                - True if the property holds, False otherwise
                - A counterexample if the property doesn't hold, None otherwise
        """
        # Ensure the model is encoded
        if len(self.solver.assertions()) == 0:
            self.encode_rbac_model()
            
        # Check the negation of the property
        self.solver.push()
        self.solver.add(Not(property_formula))
        
        result = self.solver.check()
        
        if result == sat:
            # Property doesn't hold, get counterexample
            model = self.solver.model()
            counterexample = self._extract_counterexample(model)
            self.solver.pop()
            return False, counterexample
        else:
            # Property holds
            self.solver.pop()
            return True, None
    
    def _extract_counterexample(self, model):
        """
        Extract a human-readable counterexample from a Z3 model.
        """
        counterexample = {
            "user_roles": {},
            "role_permissions": {},
            "permission_resources": {},
            "role_hierarchy": {}
        }
        
        # Extract user-role assignments
        for user in self.users:
            counterexample["user_roles"][user] = []
            for role in self.roles:
                if is_true(model.evaluate(self.z3_user_role[user][role])):
                    counterexample["user_roles"][user].append(role)
                    
        # Extract role-permission assignments
        for role in self.roles:
            counterexample["role_permissions"][role] = []
            for permission in self.permissions:
                if is_true(model.evaluate(self.z3_role_perm[role][permission])):
                    counterexample["role_permissions"][role].append(permission)
                    
        # Extract permission-resource assignments
        for permission in self.permissions:
            counterexample["permission_resources"][permission] = []
            for resource in self.resources:
                if is_true(model.evaluate(self.z3_perm_resource[permission][resource])):
                    counterexample["permission_resources"][permission].append(resource)
                    
        # Extract role hierarchy
        for senior_role in self.roles:
            counterexample["role_hierarchy"][senior_role] = []
            for junior_role in self.roles:
                if is_true(model.evaluate(self.z3_role_hierarchy[senior_role][junior_role])):
                    counterexample["role_hierarchy"][senior_role].append(junior_role)
                    
        return counterexample
    
    def verify_separation_of_duty(self, role1: str, role2: str):
        """
        Verify that no user can have both role1 and role2 (static separation of duty).
        """
        if role1 not in self.roles or role2 not in self.roles:
            raise ValueError(f"Roles must exist in the model")
            
        # Property: No user can have both roles
        property_formula = And([
            Not(And(self.z3_user_role[user][role1], self.z3_user_role[user][role2]))
            for user in self.users
        ])
        
        return self.verify_property(property_formula)
    
    def verify_least_privilege(self, role: str, required_permissions: List[str]):
        """
        Verify that a role has exactly the required permissions, no more and no less.
        """
        if role not in self.roles:
            raise ValueError(f"Role '{role}' does not exist")
        for perm in required_permissions:
            if perm not in self.permissions:
                raise ValueError(f"Permission '{perm}' does not exist")
                
        # Property: Role has exactly the required permissions
        has_required = And([
            self.z3_role_perm[role][perm] for perm in required_permissions
        ])
        
        doesnt_have_others = And([
            Not(self.z3_role_perm[role][perm]) 
            for perm in self.permissions if perm not in required_permissions
        ])
        
        property_formula = And(has_required, doesnt_have_others)
        
        return self.verify_property(property_formula)
    
    def verify_role_containment(self, containing_role: str, contained_role: str):
        """
        Verify that one role contains all permissions of another role.
        """
        if containing_role not in self.roles or contained_role not in self.roles:
            raise ValueError(f"Roles must exist in the model")
            
        # Property: For every permission that contained_role has, containing_role also has it
        property_formula = And([
            Implies(
                self.z3_role_perm[contained_role][perm],
                self.z3_role_perm[containing_role][perm]
            )
            for perm in self.permissions
        ])
        
        return self.verify_property(property_formula)
    
    def verify_no_access(self, user: str, resource: str):
        """
        Verify that a user cannot access a specific resource.
        """
        if user not in self.users:
            raise ValueError(f"User '{user}' does not exist")
        if resource not in self.resources:
            raise ValueError(f"Resource '{resource}' does not exist")
            
        # Property: User cannot access the resource through any role and permission
        property_formula = Not(Or([
            And(
                self.z3_user_role[user][role],
                self.z3_role_perm[role][perm],
                self.z3_perm_resource[perm][resource]
            )
            for role in self.roles
            for perm in self.permissions
        ]))
        
        return self.verify_property(property_formula)
    
    def verify_resource_isolation(self, resource1: str, resource2: str):
        """
        Verify that no user can access both resource1 and resource2.
        """
        if resource1 not in self.resources or resource2 not in self.resources:
            raise ValueError(f"Resources must exist in the model")
            
        # Property: No user can access both resources
        property_formula = And([
            Not(And(
                Or([
                    And(
                        self.z3_user_role[user][role1],
                        self.z3_role_perm[role1][perm1],
                        self.z3_perm_resource[perm1][resource1]
                    )
                    for role1 in self.roles
                    for perm1 in self.permissions
                ]),
                Or([
                    And(
                        self.z3_user_role[user][role2],
                        self.z3_role_perm[role2][perm2],
                        self.z3_perm_resource[perm2][resource2]
                    )
                    for role2 in self.roles
                    for perm2 in self.permissions
                ])
            ))
            for user in self.users
        ])
        
        return self.verify_property(property_formula)


def demo():
    """
    Demonstrate the RBAC model with formal verification.
    """
    # Create an RBAC model
    rbac = RBACModel()
    
    # Add users
    rbac.add_user("alice")
    rbac.add_user("bob")
    rbac.add_user("charlie")
    rbac.add_user("dave")
    
    # Add roles
    rbac.add_role("admin")
    rbac.add_role("manager")
    rbac.add_role("developer")
    rbac.add_role("user")
    
    # Add permissions
    rbac.add_permission("read")
    rbac.add_permission("write")
    rbac.add_permission("execute")
    rbac.add_permission("delete")
    rbac.add_permission("create")
    
    # Add resources
    rbac.add_resource("file_system")
    rbac.add_resource("database")
    rbac.add_resource("network")
    rbac.add_resource("application")
    
    # Set up role hierarchy
    rbac.add_role_hierarchy("admin", "manager")
    rbac.add_role_hierarchy("manager", "developer")
    rbac.add_role_hierarchy("developer", "user")
    
    # Assign users to roles
    rbac.assign_user_to_role("alice", "admin")
    rbac.assign_user_to_role("bob", "manager")
    rbac.assign_user_to_role("charlie", "developer")
    rbac.assign_user_to_role("dave", "user")
    
    # Assign permissions to roles
    rbac.assign_permission_to_role("read", "user")
    rbac.assign_permission_to_role("write", "developer")
    rbac.assign_permission_to_role("execute", "developer")
    rbac.assign_permission_to_role("delete", "manager")
    rbac.assign_permission_to_role("create", "admin")
    
    # Assign permissions to resources
    rbac.assign_permission_to_resource("read", "file_system")
    rbac.assign_permission_to_resource("read", "database")
    rbac.assign_permission_to_resource("write", "file_system")
    rbac.assign_permission_to_resource("execute", "application")
    rbac.assign_permission_to_resource("delete", "database")
    rbac.assign_permission_to_resource("create", "network")
    
    # Print the RBAC model
    print("=== RBAC Model ===")
    print(f"Users: {rbac.users}")
    print(f"Roles: {rbac.roles}")
    print(f"Permissions: {rbac.permissions}")
    print(f"Resources: {rbac.resources}")
    
    print("\n=== Role Hierarchy ===")
    for role, junior_roles in rbac.role_hierarchy.items():
        if junior_roles:
            print(f"{role} inherits from: {junior_roles}")
    
    print("\n=== User-Role Assignments ===")
    for user, roles in rbac.user_role_assignments.items():
        print(f"{user} has roles: {roles}")
        print(f"{user} has effective roles: {rbac.get_all_roles_for_user(user)}")
    
    print("\n=== Role-Permission Assignments ===")
    for role in rbac.roles:
        print(f"{role} has direct permissions: {rbac.role_permission_assignments[role]}")
        print(f"{role} has effective permissions: {rbac.get_all_permissions_for_role(role)}")
    
    print("\n=== Permission-Resource Assignments ===")
    for permission, resources in rbac.permission_resource_assignments.items():
        print(f"{permission} applies to resources: {resources}")
    
    print("\n=== Access Checks ===")
    users = list(rbac.users)
    resources = list(rbac.resources)
    
    for user in users:
        for resource in resources:
            access = rbac.check_user_access_to_resource(user, resource)
            print(f"{user} can access {resource}: {access}")
    
    # Formal verification
    print("\n=== Formal Verification ===")
    
    # Encode the RBAC model
    rbac.encode_rbac_model()
    
    # Verify separation of duty
    print("\n1. Verify Separation of Duty (admin and user roles)")
    result, counterexample = rbac.verify_separation_of_duty("admin", "user")
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")
    
    # Verify least privilege
    print("\n2. Verify Least Privilege (developer role)")
    result, counterexample = rbac.verify_least_privilege("developer", ["write", "execute"])
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")
    
    # Verify role containment
    print("\n3. Verify Role Containment (admin contains manager)")
    result, counterexample = rbac.verify_role_containment("admin", "manager")
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")
    
    # Verify no access
    print("\n4. Verify No Access (dave to network)")
    result, counterexample = rbac.verify_no_access("dave", "network")
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")
    
    # Verify resource isolation
    print("\n5. Verify Resource Isolation (network and database)")
    result, counterexample = rbac.verify_resource_isolation("network", "database")
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")
    
    # Custom property: Verify that all users can access at least one resource
    print("\n6. Custom Property: All users can access at least one resource")
    
    custom_property = And([
        Or([
            Or([
                And(
                    rbac.z3_user_role[user][role],
                    rbac.z3_role_perm[role][perm],
                    rbac.z3_perm_resource[perm][resource]
                )
                for role in rbac.roles
                for perm in rbac.permissions
            ])
            for resource in rbac.resources
        ])
        for user in rbac.users
    ])
    
    result, counterexample = rbac.verify_property(custom_property)
    print(f"Property holds: {result}")
    if not result:
        print(f"Counterexample: {counterexample}")


if __name__ == "__main__":
    demo()
