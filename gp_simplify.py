"""
GP Tree Algebraic Simplification using SymPy.

Converts a DEAP GP tree into a SymPy expression, simplifies it,
and optionally converts back to a DEAP tree string.

GP-specific operators (CMP, IFLTE, SIGN, MIN, MAX, etc.) are represented
as opaque SymPy Functions so the surrounding arithmetic can be simplified
without needing to understand the conditional semantics.

Usage:
    from gp_simplify import simplify_tree, simplify_tree_str

    # From a DEAP tree object
    simplified_expr, simplified_str = simplify_tree(tree, pset)

    # From a string expression
    simplified_str = simplify_tree_str(
        "SUB(SUB(SUB(x, CMP(y, z)), ONE), TWO)",
        feature_names=["x", "y", "z"],
    )
"""

import re
from typing import List, Dict, Optional, Tuple, Any

try:
    import sympy
    from sympy import (
        Symbol,
        Function,
        simplify,
        expand,
        collect,
        factor,
        Rational,
        Float,
        Number,
        Add,
        Mul,
        Pow,
    )

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


# =========================================================================
# GP-specific operators as SymPy Functions
# =========================================================================

if HAS_SYMPY:
    # Conditional / decision operators — treated as opaque black boxes
    # SymPy won't try to simplify inside these, but will simplify
    # arithmetic expressions that contain them.
    class CMP(Function):
        """Ternary comparison: -1 if a<b, 0 if a==b, 1 if a>b"""

        nargs = 2

    class SIGN(Function):
        """Sign function: -1 if x<0, else 1"""

        nargs = 1

    class IFLTE(Function):
        """If a <= b then c else d"""

        nargs = 4

    class IFLT(Function):
        """If a < b then c else d"""

        nargs = 4

    class IF(Function):
        """If cond > 0.5 then a else b"""

        nargs = 3

    class MIN(Function):
        """Minimum of two values"""

        nargs = 2

    class MAX(Function):
        """Maximum of two values"""

        nargs = 2

    class RELU(Function):
        """ReLU: max(0, x)"""

        nargs = 1

    class LRELU(Function):
        """Leaky ReLU"""

        nargs = 1

    class SIG(Function):
        """Sigmoid"""

        nargs = 1

    class TANH(Function):
        """Hyperbolic tangent"""

        nargs = 1

    class LT(Function):
        """Less than: 1 if a<b else 0"""

        nargs = 2

    class LTE(Function):
        """Less than or equal: 1 if a<=b else 0"""

        nargs = 2

    class GT(Function):
        """Greater than: 1 if a>b else 0"""

        nargs = 2

    class GTE(Function):
        """Greater than or equal: 1 if a>=b else 0"""

        nargs = 2

    # Map from GP primitive names to SymPy representations.
    # Arithmetic ops map to native SymPy; others map to opaque Functions.
    # NOTE: DIV is opaque because GP uses protected division (returns 1.0
    # when divisor is near-zero). SymPy would flatten DIV(DIV(a,b),c) into
    # a/(b*c), which changes semantics when b≈0.

    class DIV(Function):
        """Protected division — opaque to prevent chain flattening"""

        nargs = 2

    GP_TO_SYMPY_FUNCTIONS = {
        # Opaque GP-specific functions
        "DIV": DIV,
        "CMP": CMP,
        "SIGN": SIGN,
        "IFLTE": IFLTE,
        "IFLT": IFLT,
        "IF": IF,
        "MIN": MIN,
        "MAX": MAX,
        "RELU": RELU,
        "LRELU": LRELU,
        "SIG": SIG,
        "TANH": TANH,
        "LT": LT,
        "LTE": LTE,
        "GT": GT,
        "GTE": GTE,
    }


# =========================================================================
# Parsing: DEAP tree string -> SymPy expression
# =========================================================================


def _tokenize(expr_str: str) -> List[str]:
    """
    Tokenize a DEAP tree string into a list of tokens.

    Handles: function names, parentheses, commas, numbers (including
    negative and scientific notation), and terminal names.
    """
    tokens = []
    i = 0
    s = expr_str.strip()
    while i < len(s):
        if s[i] in "(),":
            tokens.append(s[i])
            i += 1
        elif s[i].isspace():
            i += 1
        else:
            # Read a token: could be a name, number, or negative number
            j = i
            # Handle negative numbers: -3.14, -1e-5
            if (
                s[i] == "-"
                and j + 1 < len(s)
                and (s[j + 1].isdigit() or s[j + 1] == ".")
            ):
                j += 1
            while j < len(s) and s[j] not in "(), \t":
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_expr(
    tokens: List[str], pos: int, symbols: Dict[str, Any]
) -> Tuple[Any, int]:
    """
    Recursive descent parser for DEAP tree expressions.

    Returns (sympy_expr, new_position).
    """
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    token = tokens[pos]

    # Check if this is a function call: NAME(arg1, arg2, ...)
    if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        func_name = token
        pos += 2  # skip name and '('

        # Parse arguments
        args = []
        while tokens[pos] != ")":
            arg, pos = _parse_expr(tokens, pos, symbols)
            args.append(arg)
            if pos < len(tokens) and tokens[pos] == ",":
                pos += 1  # skip comma

        pos += 1  # skip ')'

        # Convert to SymPy
        return _apply_function(func_name, args, symbols), pos

    else:
        # Terminal: number or symbol name
        pos += 1
        return _parse_terminal(token, symbols), pos


def _parse_terminal(token: str, symbols: Dict[str, Any]) -> Any:
    """Convert a terminal token to a SymPy expression."""
    # Try as a number first
    try:
        val = float(token)
        # Use Rational for exact integers, Float for others
        if val == int(val) and abs(val) < 1e10:
            return sympy.Integer(int(val))
        else:
            return sympy.Float(val, 15)
    except ValueError:
        pass

    # It's a symbol name — get or create it
    if token not in symbols:
        symbols[token] = Symbol(token)
    return symbols[token]


def _apply_function(name: str, args: list, symbols: Dict[str, Any]) -> Any:
    """
    Apply a GP function to SymPy arguments.

    Arithmetic operations are converted to native SymPy for simplification.
    GP-specific operations are kept as opaque Function calls.
    """
    # Native SymPy arithmetic
    if name == "ADD":
        return args[0] + args[1]
    elif name == "SUB":
        return args[0] - args[1]
    elif name == "MUL":
        return args[0] * args[1]
    elif name == "DIV":
        # Protected division — keep opaque to prevent chain flattening.
        # DIV(DIV(a,b),c) must NOT become a/(b*c) because protected
        # division returns 1.0 for near-zero divisors.
        # But we can fold when the divisor is a known nonzero constant.
        if args[1].is_number and abs(float(args[1])) > 1e-10:
            return args[0] / args[1]
        elif args[1] == 0:
            return sympy.Integer(1)
        else:
            return GP_TO_SYMPY_FUNCTIONS["DIV"](args[0], args[1])
    elif name == "NEG":
        return -args[0]
    elif name == "ABS":
        return sympy.Abs(args[0])
    elif name == "SQRT":
        return sympy.sqrt(sympy.Abs(args[0]))
    elif name == "SQUARE":
        return args[0] ** 2
    elif name == "POW":
        return args[0] ** args[1]
    elif name == "SIN":
        return sympy.sin(args[0])
    elif name == "COS":
        return sympy.cos(args[0])
    elif name == "EXP":
        return sympy.exp(args[0])
    elif name == "LOG":
        return sympy.log(args[0])

    # GP-specific opaque functions
    elif name in GP_TO_SYMPY_FUNCTIONS:
        func_class = GP_TO_SYMPY_FUNCTIONS[name]
        return func_class(*args)

    else:
        # Unknown function — create a generic SymPy Function
        generic = Function(name)
        return generic(*args)


def parse_gp_expression(
    expr_str: str,
    feature_names: Optional[List[str]] = None,
    constant_map: Optional[Dict[str, float]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Parse a DEAP GP tree string into a SymPy expression.

    Args:
        expr_str: The tree as a string, e.g. "ADD(MUL(x, TWO), y)"
        feature_names: Known feature names (created as Symbols)
        constant_map: Map of constant names to their numeric values.
                      If provided, constants are substituted with their
                      numeric values before simplification.

    Returns:
        (sympy_expr, symbol_dict)
    """
    if not HAS_SYMPY:
        raise ImportError("sympy is required for expression simplification")

    symbols = {}
    if feature_names:
        for name in feature_names:
            symbols[name] = Symbol(name)

    tokens = _tokenize(expr_str)
    expr, final_pos = _parse_expr(tokens, 0, symbols)

    # Substitute known constants with their numeric values
    if constant_map:
        subs = {}
        for name, value in constant_map.items():
            if name in symbols:
                if value == int(value) and abs(value) < 1e10:
                    subs[symbols[name]] = sympy.Integer(int(value))
                else:
                    subs[symbols[name]] = sympy.Float(value, 6)
        if subs:
            expr = expr.subs(subs)

    return expr, symbols


# =========================================================================
# SymPy expression -> GP tree string
# =========================================================================


def _sympy_to_gp_str(
    expr, depth: int = 0, const_collector: Optional[Dict] = None
) -> str:
    """
    Convert a SymPy expression back to a DEAP-compatible GP tree string.

    Numeric literals are emitted as named constants (_K0, _K1, ...) and
    recorded in const_collector so the caller can register them in the pset.
    If const_collector is None, bare numeric strings are emitted instead
    (suitable for display but not for round-tripping to DEAP).

    Args:
        expr: SymPy expression
        depth: Current recursion depth
        const_collector: If provided, a dict that will be populated with
                         {name: value} for each numeric constant encountered.
                         Pass an empty dict {} to enable collection.
    """
    if isinstance(expr, Symbol):
        return str(expr)

    if isinstance(expr, (sympy.Integer, sympy.Float, sympy.Rational)):
        val = float(expr)
        if const_collector is not None:
            # Emit a named constant reference
            name = _get_const_name(val, const_collector)
            return name
        else:
            # Display mode: emit bare numeric string
            if val == int(val) and abs(val) < 1e10:
                return str(int(val))
            return f"{val:.6g}"

    if isinstance(expr, sympy.NumberSymbol):
        val = float(expr)
        if const_collector is not None:
            name = _get_const_name(val, const_collector)
            return name
        return f"{val:.6g}"

    # Handle negation: -x -> NEG(x)
    if isinstance(expr, Mul) and len(expr.args) == 2:
        if expr.args[0] == -1:
            return f"NEG({_sympy_to_gp_str(expr.args[1], depth+1, const_collector)})"

    # Handle Add: a + b + c -> ADD(ADD(a, b), c)
    if isinstance(expr, Add):
        args = list(expr.args)
        result = _sympy_to_gp_str(args[0], depth + 1, const_collector)
        for arg in args[1:]:
            # Check if this term is negative (only if leading factor is numeric)
            is_negative = False
            if isinstance(arg, Mul) and arg.args[0].is_number:
                try:
                    is_negative = arg.args[0] < 0
                except TypeError:
                    pass
            if is_negative:
                pos_arg = -arg
                result = f"SUB({result}, {_sympy_to_gp_str(pos_arg, depth+1, const_collector)})"
            else:
                result = (
                    f"ADD({result}, {_sympy_to_gp_str(arg, depth+1, const_collector)})"
                )
        return result

    # Handle Mul: a * b * c -> MUL(MUL(a, b), c)
    if isinstance(expr, Mul):
        args = list(expr.args)
        # Pull out -1 coefficient
        if args[0] == -1:
            inner = Mul(*args[1:]) if len(args) > 2 else args[1]
            return f"NEG({_sympy_to_gp_str(inner, depth+1, const_collector)})"

        result = _sympy_to_gp_str(args[0], depth + 1, const_collector)
        for arg in args[1:]:
            if isinstance(arg, Pow) and arg.args[1] == -1:
                result = f"DIV({result}, {_sympy_to_gp_str(arg.args[0], depth+1, const_collector)})"
            else:
                result = (
                    f"MUL({result}, {_sympy_to_gp_str(arg, depth+1, const_collector)})"
                )
        return result

    # Handle Pow
    # Always decompose into MUL/DIV chains since SQUARE, SQRT, and POW
    # may not be registered in the evolution pset.
    if isinstance(expr, Pow):
        base, exp = expr.args
        base_str = lambda: _sympy_to_gp_str(base, depth + 1, const_collector)
        one_str = lambda: _sympy_to_gp_str(sympy.Integer(1), depth + 1, const_collector)

        # Integer exponents: decompose into MUL/DIV chains
        if exp.is_integer:
            n = int(exp)
            if n == 0:
                return one_str()
            elif n > 0:
                # x^n = MUL(x, MUL(x, ...))
                result = base_str()
                for _ in range(n - 1):
                    result = f"MUL({result}, {base_str()})"
                return result
            else:
                # x^(-n) = DIV(1, x^n)
                inner = base_str()
                for _ in range(-n - 1):
                    inner = f"MUL({inner}, {base_str()})"
                return f"DIV({one_str()}, {inner})"

        # Rational exponents: try to decompose
        if hasattr(exp, "p") and hasattr(exp, "q"):
            # SymPy Rational: exp = p/q
            p, q = int(exp.p), int(exp.q)
            # For half-integer exponents, fall through to POW
            # since we can't express sqrt with just MUL/DIV

        # Fallback: emit POW (may fail if POW not in pset, but this
        # only triggers for non-integer exponents like x^(1/2))
        return f"POW({base_str()}, {_sympy_to_gp_str(exp, depth+1, const_collector)})"

    # Handle Abs
    if isinstance(expr, sympy.Abs):
        return f"ABS({_sympy_to_gp_str(expr.args[0], depth+1, const_collector)})"

    # Handle trig/exp/log
    if isinstance(expr, sympy.sin):
        return f"SIN({_sympy_to_gp_str(expr.args[0], depth+1, const_collector)})"
    if isinstance(expr, sympy.cos):
        return f"COS({_sympy_to_gp_str(expr.args[0], depth+1, const_collector)})"
    if isinstance(expr, sympy.exp):
        return f"EXP({_sympy_to_gp_str(expr.args[0], depth+1, const_collector)})"
    if isinstance(expr, sympy.log):
        return f"LOG({_sympy_to_gp_str(expr.args[0], depth+1, const_collector)})"

    # Handle GP-specific opaque Functions
    func_name = type(expr).__name__
    if hasattr(expr, "args") and func_name in GP_TO_SYMPY_FUNCTIONS:
        arg_strs = [_sympy_to_gp_str(a, depth + 1, const_collector) for a in expr.args]
        return f"{func_name}({', '.join(arg_strs)})"

    # Generic function
    if isinstance(expr, sympy.Function):
        func_name = type(expr).__name__
        arg_strs = [_sympy_to_gp_str(a, depth + 1, const_collector) for a in expr.args]
        return f"{func_name}({', '.join(arg_strs)})"

    # Fallback
    return str(expr)


def _get_const_name(value: float, const_collector: Dict[str, float]) -> str:
    """
    Get or create a named constant for a numeric value.

    Reuses existing names if the same value has been seen before.
    Names are like _K0, _K1, etc.
    """
    # Check if this value already has a name
    for name, val in const_collector.items():
        if abs(val - value) < 1e-12:
            return name

    idx = len(const_collector)
    name = f"_K{idx}"
    const_collector[name] = value
    return name


# =========================================================================
# Main Simplification API
# =========================================================================


def simplify_expression(
    expr_str: str,
    feature_names: Optional[List[str]] = None,
    constant_map: Optional[Dict[str, float]] = None,
    collect_features: bool = True,
    for_round_trip: bool = False,
) -> Tuple[str, Any, Dict]:
    """
    Parse a GP tree string, simplify it algebraically, and return
    both the simplified string and SymPy expression.

    Args:
        expr_str: DEAP tree string, e.g. "SUB(SUB(ADD(x, ONE), y), TWO)"
        feature_names: List of feature terminal names
        constant_map: Dict mapping constant names to numeric values.
                      These get substituted before simplification, collapsing
                      constant subexpressions.
        collect_features: If True, collect terms by feature symbols for
                          a cleaner representation.

    Returns:
        (simplified_gp_str, simplified_sympy_expr, symbols_dict)
        or if for_round_trip=True:
        (simplified_gp_str, simplified_sympy_expr, symbols_dict, const_collector)

    Example:
        >>> simplify_expression(
        ...     "SUB(SUB(ADD(MUL(TWO, x), y), ONE), TWO)",
        ...     feature_names=["x", "y"],
        ...     constant_map={"TWO": 2.0, "ONE": 1.0},
        ... )
        ('SUB(ADD(MUL(2, x), y), 3)', 2*x + y - 3, {...})
    """
    if not HAS_SYMPY:
        raise ImportError("sympy is required. Install with: pip install sympy")

    # Parse to SymPy
    expr, symbols = parse_gp_expression(expr_str, feature_names, constant_map)

    # Simplify — use progressive fallbacks since sympy.simplify can fail
    # on expressions with opaque functions (CMP, MAX, etc.) that trigger
    # relational checks SymPy can't resolve with symbolic variables.
    simplified = expr
    try:
        simplified = sympy.simplify(expr)
    except (TypeError, ValueError):
        # sympy.simplify failed — try lighter-weight alternatives
        try:
            simplified = sympy.cancel(expr)
        except Exception:
            try:
                simplified = sympy.expand(expr)
            except Exception:
                pass  # Keep the parsed (constant-substituted) expression as-is

    # Expand and re-simplify to catch more opportunities
    try:
        expanded = sympy.expand(simplified)
        # If expanding made it simpler (fewer operations), use that
        if _count_ops(expanded) < _count_ops(simplified):
            simplified = expanded
    except (TypeError, ValueError, Exception):
        pass

    # Optionally collect by feature symbols for readability
    if collect_features and feature_names:
        feature_syms = [symbols[n] for n in feature_names if n in symbols]
        if feature_syms:
            try:
                simplified = collect(simplified, feature_syms)
            except Exception:
                pass  # collect can fail on complex expressions; that's fine

    # Convert back to GP string
    # Use const_collector for round-trip-safe output if requested
    if for_round_trip:
        const_collector = {}
        gp_str = _sympy_to_gp_str(simplified, const_collector=const_collector)
        return gp_str, simplified, symbols, const_collector
    else:
        gp_str = _sympy_to_gp_str(simplified)
        return gp_str, simplified, symbols


def _count_ops(expr) -> int:
    """Count the number of operations in a SymPy expression."""
    if isinstance(expr, (Symbol, sympy.Number)):
        return 0
    if hasattr(expr, "args"):
        return 1 + sum(_count_ops(a) for a in expr.args)
    return 0


def simplify_tree(tree, pset, constant_map: Optional[Dict[str, float]] = None):
    """
    Simplify a DEAP GP tree.

    Args:
        tree: A DEAP PrimitiveTree
        pset: The PrimitiveSet used
        constant_map: Dict of constant names to values. If None, attempts
                      to extract from the pset's terminals.

    Returns:
        (simplified_gp_str, simplified_sympy_expr, original_str)
    """
    expr_str = str(tree)

    # Extract feature names from pset
    if hasattr(pset, "arguments"):
        feature_names = list(pset.arguments)
    else:
        feature_names = None

    # Extract constants from pset if not provided
    if constant_map is None:
        constant_map = _extract_constants_from_pset(pset)

    gp_str, sympy_expr, symbols = simplify_expression(
        expr_str,
        feature_names=feature_names,
        constant_map=constant_map,
    )

    return gp_str, sympy_expr, expr_str


def simplify_to_primitive_tree(tree, pset, constant_map=None):
    """
    Simplify a DEAP GP tree and return a new PrimitiveTree.

    The simplified expression may contain folded numeric constants that
    aren't in the original pset. This function handles that by creating
    a copy of the pset with the necessary constant terminals registered,
    then parsing the simplified string into a PrimitiveTree.

    Compatible with both regular DEAP psets and tracked (FCP) psets.

    Args:
        tree: A DEAP PrimitiveTree
        pset: The PrimitiveSet used
        constant_map: Dict of constant names to values. If None, attempts
                      to auto-extract from the pset.

    Returns:
        (new_tree, new_pset, sympy_expr, original_str)

        new_tree: A PrimitiveTree representing the simplified expression.
                  Can be used with evaluate_with_tracking, gp.compile, etc.
        new_pset: A copy of the original pset with any new folded-constant
                  terminals registered (needed to compile new_tree).
        sympy_expr: The SymPy representation for display.
        original_str: The original tree string for comparison.
    """
    from copy import deepcopy
    from deap import gp as deap_gp

    original_str = str(tree)

    # Extract feature names from pset
    if hasattr(pset, "arguments"):
        feature_names = list(pset.arguments)
    else:
        feature_names = None

    # Extract constants from pset if not provided
    if constant_map is None:
        constant_map = _extract_constants_from_pset(pset)

    # Simplify with round-trip mode to get named constants
    gp_str, sympy_expr, symbols, const_collector = simplify_expression(
        original_str,
        feature_names=feature_names,
        constant_map=constant_map,
        for_round_trip=True,
    )

    # Build new pset with folded constants registered
    new_pset = deepcopy(pset)

    # Check if the pset uses TrackedValue terminals (for FCP compatibility)
    uses_tracked = False
    if hasattr(new_pset, "context"):
        for name, val in new_pset.context.items():
            if hasattr(val, "signed_contributions"):
                uses_tracked = True
                break

    # Register each folded constant
    for name, val in const_collector.items():
        if uses_tracked:
            try:
                from fcp_tracker import TrackedValue

                tracked_val = TrackedValue.from_constant(val, name)
                new_pset.addTerminal(tracked_val, name=name)
            except ImportError:
                new_pset.addTerminal(val, name=name)
        else:
            new_pset.addTerminal(val, name=name)

    # Parse into a PrimitiveTree
    try:
        new_tree = deap_gp.PrimitiveTree.from_string(gp_str, new_pset)
    except Exception as e:
        raise ValueError(
            f"Failed to parse simplified expression back to PrimitiveTree.\n"
            f"Simplified string: {gp_str}\n"
            f"Error: {e}\n"
            f"This can happen if SymPy produced an expression using operators "
            f"not in the primitive set. Try providing a more complete constant_map."
        ) from e

    return new_tree, new_pset, sympy_expr, original_str


def _extract_constants_from_pset(pset) -> Dict[str, float]:
    """
    Extract constant terminal values from a DEAP PrimitiveSet.

    DEAP stores actual terminal objects (including TrackedValues) in
    pset.context, while pset.terminals only has string references.
    This function checks both locations.
    """
    constants = {}

    # Get the set of feature argument names to exclude
    feature_names = set()
    if hasattr(pset, "arguments"):
        feature_names = set(pset.arguments)

    # Primary source: pset.context has the actual values
    if hasattr(pset, "context"):
        for name, val in pset.context.items():
            if name in feature_names:
                continue
            # Skip builtins and callable primitives
            if name.startswith("__") or callable(val) and not hasattr(val, "value"):
                continue

            # Handle TrackedValue objects
            if hasattr(val, "value") and hasattr(val, "signed_contributions"):
                constants[name] = float(val.value)
            elif isinstance(val, (int, float)):
                constants[name] = float(val)

    # Fallback: also check pset.terminals for plain numeric terminals
    if not constants and hasattr(pset, "terminals"):
        for type_key, term_list in pset.terminals.items():
            for term in term_list:
                name = term.name if hasattr(term, "name") else str(term)
                if name in feature_names:
                    continue

                val = term.value if hasattr(term, "value") else None
                if val is None:
                    continue
                if hasattr(val, "value"):
                    val = val.value
                if isinstance(val, (int, float)):
                    constants[name] = float(val)

    return constants


# =========================================================================
# Pretty Printing
# =========================================================================


def expression_summary(
    expr_str: str,
    feature_names: Optional[List[str]] = None,
    constant_map: Optional[Dict[str, float]] = None,
) -> str:
    """
    Return a human-readable summary comparing original and simplified.

    Useful for printing in notebooks.
    """
    gp_str, sympy_expr, symbols = simplify_expression(
        expr_str, feature_names, constant_map
    )

    original_ops = expr_str.count("(")
    simplified_ops = gp_str.count("(")
    reduction = (1 - simplified_ops / original_ops) * 100 if original_ops > 0 else 0

    lines = [
        "GP Expression Simplification",
        "=" * 50,
        f"Original ({original_ops} ops):",
        f"  {expr_str}",
        "",
        f"Simplified ({simplified_ops} ops, {reduction:.0f}% reduction):",
        f"  {gp_str}",
        "",
        f"SymPy form:",
        f"  {sympy_expr}",
    ]

    return "\n".join(lines)


# =========================================================================
# Example / Test
# =========================================================================

if __name__ == "__main__":
    if not HAS_SYMPY:
        print("sympy not installed. Install with: pip install sympy")
        exit(1)

    print("GP Tree Simplification Demo")
    print("=" * 60)

    # Test 1: Constant folding
    print("\nTest 1: Constant folding")
    result = expression_summary(
        "SUB(SUB(ADD(MUL(TWO, x), y), ONE), TWO)",
        feature_names=["x", "y"],
        constant_map={"TWO": 2.0, "ONE": 1.0},
    )
    print(result)

    # Test 2: Nested SUBs collapse
    print("\nTest 2: Nested SUB chain")
    result = expression_summary(
        "SUB(SUB(SUB(x, y), z), w)",
        feature_names=["x", "y", "z", "w"],
    )
    print(result)

    # Test 3: Expression with CMP (opaque function preserved)
    print("\nTest 3: Arithmetic around CMP")
    result = expression_summary(
        "SUB(ADD(x, NEG(CMP(y, z))), MUL(TWO, x))",
        feature_names=["x", "y", "z"],
        constant_map={"TWO": 2.0},
    )
    print(result)

    # Test 4: Redundant operations
    print("\nTest 4: ADD(x, NEG(x)) = 0")
    result = expression_summary(
        "ADD(x, NEG(x))",
        feature_names=["x"],
    )
    print(result)

    # Test 5: Complex example mimicking your tree
    print("\nTest 5: Complex concrete example")
    result = expression_summary(
        "SUB(SUB(SUB(SUB(SUB(MUL(C0, x), CMP(y, z)), DIV(w, NEG(C1))), C2), CMP(C3, v)), NEG(CMP(u, SUB(C4, x))))",
        feature_names=["x", "y", "z", "w", "v", "u"],
        constant_map={"C0": 10.0, "C1": -4.5, "C2": 3.0, "C3": 50.0, "C4": 30.0},
    )
    print(result)
