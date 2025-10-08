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
            epilog="""任务说明:
  occurrence  从蛋白质结构预测药物-靶标结合发生
  site        从实验蛋白质结构预测残基水平结合位点
  affinity    从蛋白质-配体复合物结构预测结合亲和力

使用示例:
  python DTBind_test.py occurrence
  python DTBind_test.py site
  python DTBind_test.py affinity
            """
        )

        # 必需参数
        parser.add_argument(
            'task',
            choices=['occurrence', 'site', 'affinity'],
            help='选择预测任务类型'
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
                print(f"\n {config['description']} 预测完成!")
            else:
                print(f"\n 预测过程出现错误")
                sys.exit(result.returncode)

        except subprocess.CalledProcessError as e:
            print(f"\n 预测失败，退出码: {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n 预测被用户中断")
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
