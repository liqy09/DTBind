import argparse
import sys
import os
import subprocess


class DTBindTrainer:
    def __init__(self):
        self.task_config = {
            'occurrence': {
                'script': 'dti_train.py',
                'description': 'Drug-Target Binding Occurrence Prediction Training',
                'directory': 'script/occurrence'
            },
            'site': {
                'script': 'site_train.py',
                'description': 'Binding Site Prediction Training',
                'directory': 'script/site'
            },
            'affinity': {
                'script': 'aff_train.py',
                'description': 'Binding Affinity Prediction Training',
                'directory': 'script/affinity'
            }
        }

    def setup_parser(self):

        parser = argparse.ArgumentParser(
            description='DTBind - Unified Training Interface'
        )

        parser.add_argument(
            'task',
            choices=['occurrence', 'site', 'affinity'],
            help='Select the training task type'
        )

        return parser

    def run_training(self, task):
        config = self.task_config[task]

        # Check if the directory and script exist
        if not os.path.exists(config['directory']):
            print(f"Error: Directory '{config['directory']}' does not exist")
            sys.exit(1)

        script_path = os.path.join(config['directory'], config['script'])
        if not os.path.exists(script_path):
            print(f"Error: Script file '{script_path}' does not exist")
            sys.exit(1)

        # Change to the corresponding directory and execute the command
        original_dir = os.getcwd()
        try:
            os.chdir(config['directory'])
            cmd = ['python', config['script']]

            # Run the training script
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                sys.exit(result.returncode)

        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except FileNotFoundError:
            print(f"Error: Training script '{config['script']}' not found")
            sys.exit(1)
        except KeyboardInterrupt:
            sys.exit(1)
        finally:
            os.chdir(original_dir)

    def main(self):
        parser = self.setup_parser()
        args = parser.parse_args()

        self.run_training(args.task)


if __name__ == '__main__':
    trainer = DTBindTrainer()
    trainer.main()