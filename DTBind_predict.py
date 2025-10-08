import argparse
import sys
import os
import subprocess


class DTBindTester:
    def __init__(self):
        self.task_config = {
            'occurrence': {
                'script': 'dti_test.py',
                'description': 'Drug-Target Binding Occurrence Prediction',
                'directory': 'script/occurrence'
            },
            'site': {
                'script': 'site_test.py',
                'description': 'Binding Site Prediction',
                'directory': 'script/site'
            },
            'affinity': {
                'script': 'aff_test.py',
                'description': 'Binding Affinity Prediction',
                'directory': 'script/affinity'
            }
        }

    def setup_parser(self):
        parser = argparse.ArgumentParser(
            description='DTBind - Unified Prediction Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""Task descriptions:
  occurrence  Predict drug-target binding occurrence
  site        Predict residue-level binding sites
  affinity    Predict binding affinity

Usage examples:
  python DTBind_test.py occurrence
  python DTBind_test.py site
  python DTBind_test.py affinity
            """
        )

        # Required arguments
        parser.add_argument(
            'task',
            choices=['occurrence', 'site', 'affinity'],
            help='Select the prediction task type'
        )

        return parser

    def run_prediction(self, task):
        config = self.task_config[task]

        original_dir = os.getcwd()
        os.chdir(config['directory'])

        try:
            cmd = ['python', config['script']]
            result = subprocess.run(cmd, check=True)
            if result.returncode == 0:
                print(f"\n {config['description']} completed!")
            else:
                print(f"\n An error occurred during prediction")
                sys.exit(result.returncode)

        except subprocess.CalledProcessError as e:
            print(f"\n Prediction failed with exit code: {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n Prediction was interrupted by the user")
            sys.exit(1)
        finally:
            os.chdir(original_dir)

    def main(self):
        parser = self.setup_parser()
        args = parser.parse_args()

        self.run_prediction(args.task)


if __name__ == '__main__':
    tester = DTBindTester()
    tester.main()
