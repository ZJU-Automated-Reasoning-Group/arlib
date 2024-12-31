import argparse
import os
import os.path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_bool_argument(parser, name, default=False,
                      help=None, dest=None, **kwargs):
    """
    Add boolean option that can be turned on and off
    """
    import argparse
    dest_name = dest if dest is not None else name
    mutex_group = parser.add_mutually_exclusive_group(required=False)
    mutex_group.add_argument('--' + name, dest=dest_name, type=str2bool,
                             nargs='?', const=True, help=help,
                             metavar='BOOL', **kwargs)
    mutex_group.add_argument('--no-' + name, dest=dest_name,
                             type=lambda v: not (str2bool(v)),
                             nargs='?', const=False,
                             help=argparse.SUPPRESS, **kwargs)
    parser.set_defaults(dest_name=default)


def add_help_arg(ap):
    ap.add_argument('-h', '--help', action='help',
                    help='Print this message and exit')


def add_in_args(ap):
    ap.add_argument('in_files', metavar='FILE', help='Input file', nargs='+')
    return ap


def add_in_out_args(ap):
    add_in_args(ap)
    ap.add_argument('-o', dest='out_file',
                    metavar='FILE', help='Output file name', default=None)
    return ap


def add_tmp_dir_args(ap):
    ap.add_argument('--save-temps', '--keep-temps',
                    dest="save_temps",
                    help="Do not delete temporary files",
                    action="store_true", default=False)
    ap.add_argument('--temp-dir', dest='temp_dir', metavar='DIR',
                    help="Temporary directory", default=None)
    return ap


class CliCmd(object):
    def __init__(self, name='', help='', allow_extra=False):
        self.name = name
        self.help = help
        self.allow_extra = allow_extra

    def mk_arg_parser(self, argp):
        add_help_arg(argp)
        return argp

    def run(self, args=None, extra=[]):
        return 0

    def name_out_file(self, in_files, args=None, work_dir=None):
        out_file = 'out'
        if work_dir is not None:
            out_file = os.path.join(work_dir, out_file)
        return out_file

    def main(self, argv):
        import argparse
        ap = argparse.ArgumentParser(prog=self.name,
                                     description=self.help,
                                     add_help=False)
        ap = self.mk_arg_parser(ap)

        if self.allow_extra:
            args, extra = ap.parse_known_args(argv)
        else:
            args = ap.parse_args(argv)
            extra = []
        return self.run(args, extra)
