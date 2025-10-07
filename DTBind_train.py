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
                'directory': 'occurrence'
            },
            'site': {
                'script': 'site_train.py',
                'description': 'Binding Site Prediction Training',
                'directory': 'site'
            },
            'affinity': {
                'script': 'aff_train.py',
                'description': 'Binding Affinity Prediction Training',
                'directory': 'affinity'
            }
        }

    def setup_parser(self):
        """设置命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description='DTBind - Unified Training Interface'
        )

        parser.add_argument(
            'task',
            choices=['occurrence', 'site', 'affinity'],
            help='选择训练任务类型'
        )

        return parser

    def run_training(self, task):
        config = self.task_config[task]

        # 检查目录和脚本是否存在
        if not os.path.exists(config['directory']):
            print(f"错误: 目录 '{config['directory']}' 不存在")
            sys.exit(1)

        script_path = os.path.join(config['directory'], config['script'])
        if not os.path.exists(script_path):
            print(f"错误: 脚本文件 '{script_path}' 不存在")
            sys.exit(1)

        # 切换到对应目录并执行命令
        original_dir = os.getcwd()
        try:
            os.chdir(config['directory'])
            cmd = ['python', config['script']]

            # 执行训练脚本
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                sys.exit(result.returncode)

        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
        except FileNotFoundError:
            print(f"错误: 找不到训练脚本 '{config['script']}'")
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