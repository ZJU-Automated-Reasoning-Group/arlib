from rbac import RBACModel
from z3 import *


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
    

