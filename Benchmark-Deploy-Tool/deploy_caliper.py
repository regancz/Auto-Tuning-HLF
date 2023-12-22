import deploy_fabric


def run_caliper_and_log(ssh_client, cc_name):
    commands = ["cd /home/charles/Project/Blockchain/caliper-benchmarks && bash launch_" + cc_name + ".sh"]
    deploy_fabric.run_command_and_log(ssh_client, commands, "caliper_log")


def mv_report_and_log(ssh_client, config_id, performance_id):
    commands = [
        f"cp /home/charles/Project/Blockchain/report.html /home/charles/Project/Blockchain/caliper-benchmarks/report/report_{config_id}.html"
    ]
    deploy_fabric.run_command_and_log(ssh_client, commands, "mv_report")
