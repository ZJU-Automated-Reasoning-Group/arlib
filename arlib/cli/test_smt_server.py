#!/usr/bin/env python3
import os
import subprocess
import time
import unittest


class TestSmtServer(unittest.TestCase):

    def setUp(self):
        # Start the SMT server in a separate process before each test
        # get the current dir 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.server_process = subprocess.Popen(['python3', current_dir + '/smt_server.py'])
        time.sleep(1)  # Give the server time to start

    def tearDown(self):
        # Send exit command to server
        try:
            with open('/tmp/smt_input', 'w') as f:
                f.write('exit\n')
                f.flush()

            # Wait for a short time for graceful shutdown
            time.sleep(0.5)

            # If process is still running, terminate it
            if self.server_process.poll() is None:
                self.server_process.terminate()
                time.sleep(0.5)

                # If still running after terminate, kill it
                if self.server_process.poll() is None:
                    self.server_process.kill()

            self.server_process.wait()
        except Exception as e:
            print(f"Error during teardown: {e}")
            # Ensure process is killed even if there's an error
            try:
                self.server_process.kill()
            except:
                pass

        # Clean up the pipes
        try:
            os.remove('/tmp/smt_input')
        except:
            pass
        try:
            os.remove('/tmp/smt_output')
        except:
            pass

    def send_command(self, command: str) -> str:
        # Is this a good iea (We shoud ue IPC)?
        with open('/tmp/smt_input', 'w') as f:
            f.write(command + '\n')
            f.flush()

        with open('/tmp/smt_output', 'r') as f:
            return f.readline().strip()

    def test_declare_and_assert(self):
        # Declare variables
        response = self.send_command('declare-const x Int')
        self.assertEqual(response, 'success')

        response = self.send_command('declare-const y Int')
        self.assertEqual(response, 'success')

        # Assert constraints
        response = self.send_command('assert x > y')
        self.assertEqual(response, 'success')

        self.send_command('(push)')

        # Check satisfiability
        response = self.send_command('check-sat')
        self.assertEqual(response, 'sat')

        # Get model
        # response = self.send_command('get-value x y')
        # print(response)
        # self.assertTrue('x=' in response and 'y=' in response)
        # self.send_command('(exit)')


if __name__ == '__main__':
    unittest.main()
