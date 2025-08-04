import subprocess

# 定义 bcalm 命令
for i in [17, 18, 19, 20, 21, 22]:
    command_bcalm = [
        "/root/data1/bcalm/build/bcalm",  # bcalm 程序路径
        "-in", f"/root/data1/hg38/chr{i}_0001.fa",  # 输入文件路径
        "-kmer-size", "201",  # kmer大小
        "-abundance-min", "2"  # 最小丰度
    ]
    
    command_convertToGFA = [
        "python", "/root/data1/bcalm/scripts/convertToGFA.py",  # 执行 Python 脚本
        f"/root/data1/chr{i}_0001.unitigs.fa",  # 输入文件路径
        f"/root/data1/chr{i}_0001.unitigs.gfa",  # 输出文件路径
        "201"  # 其他参数
    ]

    try:
        # 执行 bcalm 命令
        print(f"正在运行 bcalm: chr{i}")
        result_bcalm = subprocess.run(command_bcalm, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 打印 bcalm 的输出
        print(result_bcalm.stdout)
        print(result_bcalm.stderr)

        # bcalm 执行成功，继续执行 convertToGFA 命令
        print(f"bcalm 执行成功, 正在运行 convertToGFA: chr{i}")
        result_convertToGFA = subprocess.run(command_convertToGFA, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 打印 convertToGFA 的输出
        print(result_convertToGFA.stdout)
        print(result_convertToGFA.stderr)
        print(f"convertToGFA 执行成功: chr{i}")
    
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e.cmd}")
        print(f"错误输出: {e.stderr}")
    except FileNotFoundError as e:
        print(f"未找到文件: {e}")
    except Exception as e:
        print(f"执行时发生异常: {e}")
