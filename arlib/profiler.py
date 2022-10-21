import cProfile
import glob
import logging
import multiprocessing
import os
import subprocess


class Profiler:
    """A simple wrapper around cProfile that can be used as context manager,
    honors the ``--profile`` option and automatically writes the results to a
    file. To accumulate all profiles for one process, it uses a static profiler
    and uses a filename pattern that includes the process id.
    Profiling in the main process is somewhat special, as it is enabled
    globally and should thus not be enabled again by a worker function
    that happens to be executed in the main process. Thus, to enable
    profiling in the main process, the Profiler needs to be created with
    ``is_main=True`` to confirm that the caller is aware that it is
    running within the main process. Except for one call in the main
    function, this argument should never be set to True.
    """
    enabled = True
    # static profiler, global within one process
    profiler = None

    def __init__(self, is_main=False):
        """Create a profiler."""
        if Profiler.enabled:
            if multiprocessing.parent_process() is None:
                self.enabled = True
                self.filename = '.profile.prof'
            else:
                self.enabled = True
                self.filename = f'.profile-{os.getpid()}.prof'
            if Profiler.profiler is None:
                Profiler.profiler = cProfile.Profile()

    def __enter__(self):
        """Start profiling, if ``--profile`` was given."""
        if Profiler.enabled and self.enabled:
            Profiler.profiler.enable()

    def __exit__(self, type, value, traceback):
        """Stop profiling and write results to file."""
        if Profiler.enabled and self.enabled:
            try:
                Profiler.profiler.disable()
                Profiler.profiler.dump_stats(self.filename)
            except KeyboardInterrupt as e:
                logging.warning(f'Writing {self.filename} was interrupted.'
                                'To avoid corrupted data, we remove this file.')
                os.unlink(self.filename)
                raise e


def __render_profile(infiles, dotfile, pngfile):
    """Helper method to render profile data from a set of input files to a
    png."""
    cmd = [
        'gprof2dot', '--node-label=self-time-percentage',
        '--node-label=total-time', '--node-label=total-time-percentage', '-f',
        'pstats', '-o', dotfile, *infiles
    ]
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        logging.warning('gprof2dot was not found. Try "pip install gprof2dot".')
        return False

    try:
        subprocess.run(['dot', '-Tpng', '-o', pngfile, dotfile])
    except FileNotFoundError:
        logging.warning('dot was not found. Try "apt install graphviz".')
        return False
    return True


def render_profiles():
    """Convenience function to directly render profiling data to png.
    Splits the data into two parts, the main process and the combination
    of all other processes. Uses ``gprof2dot`` and ``dot``, and will try
    to print helpful error messages if they are not available.
    """
    logging.debug('Rendering profile data')
    if __render_profile(['.profile.prof'], '.profile.dot', 'profile.png'):
        logging.info('Profile data for main process is in profile.png')

    files = [f for f in glob.glob('.*.prof') if f != '.profile.prof']
    if files:
        if __render_profile(files, '.profile-workder.dot',
                            'profile-worker.png'):
            logging.info(
                'Profile data for worker processes is in profile-worker.png')