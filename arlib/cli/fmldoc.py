"""All-in-one translator for different logic constraints?

- DIMACS
- QDIMACS
- TPLP
- FZN
- SMT-LIB2
- Sympy?
- LP
- SyGuS
- Datalog
- ...

TODO: to be tested
"""

import argparse
import sys
from typing import Optional, Sequence
from arlib.translator import (
    cnf2lp, cnf2smt, dimacs2smt, qbf2smt,
    smt2c, smt2sympy, sygus2smt, wcnf2z3
)


def get_translator(input_format: str, output_format: str):
    """Get appropriate translator function based on formats"""
    from arlib.translator import (
        cnf2lp, cnf2smt, dimacs2smt, qbf2smt,
        smt2c, smt2sympy, sygus2smt, wcnf2z3
    )

    translators = {
        ('dimacs', 'smtlib2'): dimacs2smt.translate,
        ('dimacs', 'lp'): cnf2lp.translate,
        ('dimacs', 'sympy'): lambda x: smt2sympy.translate(dimacs2smt.translate(x)),
        ('qdimacs', 'smtlib2'): qbf2smt.translate,
        ('sygus', 'smtlib2'): sygus2smt.translate,
        ('smtlib2', 'c'): smt2c.translate,
        ('smtlib2', 'sympy'): smt2sympy.translate,
        ('wcnf', 'smtlib2'): wcnf2z3.translate
    }

    return translators.get((input_format, output_format))


def detect_format(filename: str) -> str:
    """Auto-detect format from file extension"""
    ext_map = {
        '.cnf': 'dimacs',
        '.qdimacs': 'qdimacs',
        '.tplp': 'tplp',
        '.fzn': 'fzn',
        '.smt2': 'smtlib2',
        '.sy': 'sygus',
        '.lp': 'lp',
        '.dl': 'datalog'
    }

    from pathlib import Path
    ext = Path(filename).suffix.lower()
    return ext_map.get(ext)


def handle_translate(args):
    """Handle translation between formats"""
    # Auto-detect formats if requested
    input_format = args.input_format
    output_format = args.output_format

    if args.auto_detect:
        if not input_format:
            input_format = detect_format(args.input_file)
        if not output_format:
            output_format = detect_format(args.output_file)

    if not input_format or not output_format:
        raise ValueError("Input and output formats must be specified or auto-detected")

    # Get appropriate translator
    translator = get_translator(input_format, output_format)
    if not translator:
        raise ValueError(f"No translator available for {input_format} to {output_format}")

    # Read input
    with open(args.input_file) as f:
        input_content = f.read()

    # Translate
    output_content = translator(input_content)

    # Write output
    with open(args.output_file, 'w') as f:
        f.write(output_content)

    return 0


def handle_validate(args):
    """Validate file format"""
    # Read input file
    with open(args.input_file) as f:
        content = f.read()

    # Try parsing based on format
    try:
        if args.format == 'smtlib2':
            from arlib.smt.ff.ff_ir import EnhancedSMTParser
            parser = EnhancedSMTParser()
            parser.parse_smt(content)
        elif args.format == 'dimacs':
            from arlib.bool.tseitin_converter import format_formula
            # Basic validation by trying to parse
            lines = [l for l in content.splitlines() if l and not l.startswith('c')]
            if not any(l.startswith('p cnf') for l in lines):
                raise ValueError("Missing problem line")
        # Add validation for other formats

        print(f"Successfully validated {args.input_file}")
        return 0

    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


def handle_analyze(args):
    """Analyze constraint properties"""
    # Auto-detect format if not specified
    if not args.format:
        args.format = detect_format(args.input_file)
        if not args.format:
            raise ValueError("Could not detect format - please specify explicitly")

    with open(args.input_file) as f:
        content = f.read()

    # Analyze based on format
    if args.format == 'dimacs':
        # Count variables and clauses
        lines = [l for l in content.splitlines() if l and not l.startswith('c')]
        p_line = next(l for l in lines if l.startswith('p cnf'))
        num_vars, num_clauses = map(int, p_line.split()[2:4])
        print(f"Number of variables: {num_vars}")
        print(f"Number of clauses: {num_clauses}")

    elif args.format == 'smtlib2':
        from arlib.smt.ff.ff_ir import EnhancedSMTParser
        parser = EnhancedSMTParser()
        # Count declarations and assertions
        decls = len([l for l in content.splitlines() if l.strip().startswith('(declare-')])
        asserts = len([l for l in content.splitlines() if l.strip().startswith('(assert')])
        print(f"Number of declarations: {decls}")
        print(f"Number of assertions: {asserts}")

    # Add analysis for other formats

    return 0


def handle_batch(args):
    """Handle batch processing"""
    from pathlib import Path
    import os

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    success = 0
    failed = 0

    for input_file in input_dir.glob('*'):
        if input_file.is_file():
            try:
                # Auto-detect formats if needed
                in_format = args.input_format or detect_format(str(input_file))
                if not in_format:
                    continue

                out_format = args.output_format or args.input_format
                if not out_format:
                    continue

                # Construct output path
                out_ext = next(ext for ext, fmt in detect_format.__defaults__[0].items()
                               if fmt == out_format)
                output_file = output_dir / (input_file.stem + out_ext)

                # Translate
                translate_args = argparse.Namespace(
                    input_format=in_format,
                    output_format=out_format,
                    input_file=str(input_file),
                    output_file=str(output_file),
                    auto_detect=False,
                    preserve_comments=args.preserve_comments if hasattr(args, 'preserve_comments') else False
                )

                if handle_translate(translate_args) == 0:
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                failed += 1

    print(f"Batch processing complete: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Set up logging based on verbose/debug flags
    if args.debug:
        log_level = 'DEBUG'
    elif args.verbose:
        log_level = 'INFO'
    else:
        log_level = 'WARNING'

    # Execute appropriate command
    try:
        if args.command == 'translate':
            return handle_translate(args)
        elif args.command == 'validate':
            return handle_validate(args)
        elif args.command == 'analyze':
            return handle_analyze(args)
        elif args.command == 'batch':
            return handle_batch(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
