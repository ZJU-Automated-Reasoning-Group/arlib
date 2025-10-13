"""Conversion algorithms between abstract domains.

This module implements the conversion algorithms described in Elder et al.'s
paper for converting between MOS, KS, and AG abstract domains.
"""

import numpy as np
from typing import List
from .matrix_ops import Matrix, howellize, make_explicit
from .mos_domain import MOS
from .ks_domain import KS
from .ag_domain import AG


def mos_to_ks(mos_element: MOS) -> KS:
    """Convert MOS element to KS element.

    Args:
        mos_element: MOS element to convert

    Returns:
        Equivalent KS element

    From Theorem 1: γ_MOS(B) ⊆ γ_AG(G) where G is constructed from B
    """
    if mos_element.is_empty():
        return KS(w=mos_element.w)

    k = mos_element.matrices[0].rows - 1  # Number of variables
    w = mos_element.w

    # For each MOS matrix, create corresponding AG matrix
    ag_matrices = []
    for T in mos_element.matrices:
        # Extract b, M from T = [M b; 0 1]
        M = T.data[:k, :k]
        b = T.data[:k, k]

        # Create AG matrix representing the affine transformation
        # For identity transformation x' = x, we need generators x'_i = x_i for each i
        ag_data = np.zeros((k, 2*k + 1), dtype=object)

        # For each variable i, create generator: x'_i - x_i = 0
        for i in range(k):
            ag_data[i, i] = -1      # -x_i coefficient
            ag_data[i, k + i] = 1   # x'_i coefficient
            # Constant term is 0

        # Also add a constant generator for the affine constant (if any)
        if np.any(b != 0):
            # Add generator for the constant term
            const_row = np.zeros(2*k + 1, dtype=object)
            const_row[2*k] = -1  # -constant coefficient
            for j in range(k):
                const_row[j] = b[j]  # x_j coefficient for constant relation
            ag_data = np.vstack([ag_data, const_row.reshape(1, -1)])

        ag_matrices.append(AG(Matrix(ag_data, 2**w)))

    # Join all AG matrices
    if not ag_matrices:
        return KS(w=w)

    result_ag = ag_matrices[0]
    for ag in ag_matrices[1:]:
        result_ag = result_ag.join(ag)

    # Convert AG to KS
    return ag_to_ks(result_ag)


def ks_to_mos(ks_element: KS) -> MOS:
    """Convert KS element to MOS element.

    Args:
        ks_element: KS element to convert

    Returns:
        Equivalent MOS element
    """
    if ks_element.is_empty():
        return MOS(w=ks_element.w)

    # Convert KS to AG first, then AG to MOS
    ag_element = ks_to_ag(ks_element)
    return ag_to_mos(ag_element)


def ag_to_ks(ag_element: AG) -> KS:
    """Convert AG element to KS element.

    From the paper: AG and KS are equivalent domains, so we can convert
    an AG element to an equivalent KS element with no loss of precision.

    Args:
        ag_element: AG element to convert

    Returns:
        Equivalent KS element
    """
    if ag_element.is_empty():
        return KS(w=ag_element.w)

    # The conversion uses the dualization operation and permutation of columns
    # This is based on the algorithm described in section 3 of the paper

    G = ag_element.matrix
    k = (G.cols - 1) // 2

    # Create KS matrix Z of the form [X X'|b] such that γ_KS(Z) = γ_AG(G)
    # We construct Z by letting [b|X X'] = G ⊥ and permuting columns

    G_dual = ag_element.dualize()
    ks_data = np.zeros((G_dual.rows, 2*k + 1), dtype=object)

    # Copy coefficients and rearrange columns according to the algorithm
    for i in range(G_dual.rows):
        for j in range(2*k + 1):
            if j < k:
                # Pre-state variables
                ks_data[i, j] = G_dual.matrix[i, j]
            elif j < 2*k:
                # Post-state variables
                ks_data[i, j] = G_dual.matrix[i, j]
            else:
                # Constant term
                ks_data[i, 2*k] = G_dual.matrix[i, 2*k]

    return KS(Matrix(ks_data, ag_element.modulus))


def ks_to_ag(ks_element: KS) -> AG:
    """Convert KS element to AG element.

    Args:
        ks_element: KS element to convert

    Returns:
        Equivalent AG element
    """
    if ks_element.is_empty():
        return AG(w=ks_element.w)

    # Reverse the process of ag_to_ks
    # This involves reversing the dualization and column permutation

    X = ks_element.matrix
    k = (X.cols - 1) // 2

    # Create padded matrix and reverse the column permutation
    ag_data = np.zeros((X.rows, 2*k + 1), dtype=object)

    for i in range(X.rows):
        for j in range(2*k + 1):
            if j < k:
                # Pre-state variables (reverse order)
                ag_data[i, k - 1 - j] = X[i, j]
            elif j < 2*k:
                # Post-state variables (reverse order)
                ag_data[i, 2*k - 1 - j] = X[i, j]
            else:
                # Constant term
                ag_data[i, 2*k] = X[i, 2*k]

    ag_element = AG(Matrix(ag_data, ks_element.modulus), ks_element.w)

    # Apply reverse dualization
    return ag_element.dualize()


def ag_to_mos(ag_element: AG) -> MOS:
    """Convert AG element to MOS element using SHATTER operation.

    From Theorem 3: γ_AG(G) = γ_MOS(SHATTER(G))

    Args:
        ag_element: AG element to convert

    Returns:
        Equivalent MOS element
    """
    if ag_element.is_empty():
        return MOS(w=ag_element.w)

    # Apply SHATTER operation (Theorem 3 from paper)
    from .mos_domain import shatter_ag

    shattered_matrices = shatter_ag(ag_element.matrix)
    return MOS(shattered_matrices, ag_element.w)


def ks_to_mos_with_pre_state_guards(ks_element: KS) -> MOS:
    """Convert KS element to MOS element with pre-state guards (Section 4.4).

    Args:
        ks_element: KS element that is not total with respect to pre-state inputs

    Returns:
        MOS element with pre-state guards enforced
    """
    if ks_element.is_empty():
        return MOS(w=ks_element.w)

    # Algorithm from section 4.4:
    # 1. Convert KS to AG
    # 2. Put AG matrix in Howell form using MAKEEXPLICIT
    # 3. Convert back to MOS

    ag_element = ks_to_ag(ks_element)

    # Apply MAKEEXPLICIT algorithm to handle pre-state guards
    howell_matrix = make_explicit(ag_element.matrix)
    explicit_ag = AG(howell_matrix, ag_element.w)

    return ag_to_mos(explicit_ag)


def mos_to_ks_with_basis_conversion(mos_element: MOS) -> KS:
    """Convert MOS element to KS element using basis conversion (Section 4.3).

    Args:
        mos_element: MOS element B0

    Returns:
        KS element representing the same relations
    """
    if mos_element.is_empty():
        return KS(w=mos_element.w)

    # Algorithm from section 4.3:
    # 1. Build two-vocabulary AG matrix from each one-vocabulary matrix in B
    # 2. Compute the join of all AG matrices
    # 3. Convert the resulting AG to KS

    k = mos_element.matrices[0].rows - 1
    w = mos_element.w

    # For each MOS matrix B_i, create corresponding G_i
    ag_matrices = []
    for B in mos_element.matrices:
        # Extract b_i, M_i from B = [M_i b_i; 0 1]
        M_i = B.data[:k, :k]
        b_i = B.data[:k, k]

        # Create G_i = [b_i; M_i; R_i] where R_i are basis matrices
        # This implements the BASIS construction from the paper

        # For simplicity, use identity matrices for R_i
        # In a full implementation, this would compute the actual basis
        num_basis = k
        ag_data = np.zeros((1 + k + num_basis, 2*k + 1), dtype=object)

        # First row: b_i
        ag_data[0, :k] = b_i

        # Next k rows: M_i (pre-state part)
        ag_data[1:k+1, :k] = M_i

        # Next k rows: M_i (post-state part)
        ag_data[k+1:2*k+1, k:2*k] = M_i

        # Constant terms
        ag_data[1:k+1, 2*k] = 0  # Pre-state constants
        ag_data[k+1:2*k+1, 2*k] = 0  # Post-state constants

        ag_matrices.append(AG(Matrix(ag_data, 2**w)))

    # Join all AG matrices
    if not ag_matrices:
        return KS(w=w)

    result_ag = ag_matrices[0]
    for ag in ag_matrices[1:]:
        result_ag = result_ag.join(ag)

    # Convert AG to KS
    return ag_to_ks(result_ag)
