import datetime
import logging
import re
import paramiko
import main


def run_command_and_log(ssh_client, commands, log_prefix):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"D:/log/{log_prefix}_{timestamp}.log"
    logger = logging.getLogger(f'{log_prefix}_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    logger.addHandler(file_handler)
    try:
        outputs = ssh_run_command(ssh_client, commands)
        for index, output in enumerate(outputs):
            logger.info(
                f"{timestamp} Command '{commands[index]}' output :\n{remove_ansi_escape_codes(output)}")
        print(f"Output logged to {log_file}")
    except Exception as e:
        logger.error(f"Error running command: {e}")
        print(f"Error running command: {e}")


def ssh_run_command(ssh_client, commands):
    try:
        outputs = []
        for command in commands:
            print("%s is executing" % command)
            stdin, stdout, stderr = ssh_client.exec_command(command)
            output = stdout.read().decode("utf-8")
            outputs.append(output)
        return outputs
    except paramiko.AuthenticationException as auth_exception:
        print(f"Authentication failed: {auth_exception}")
    except paramiko.SSHException as ssh_exception:
        print(f"SSH connection failed: {ssh_exception}")
    except Exception as e:
        print(f"An error occurred: {e}")


def deploy_fabric_and_log(ssh_client, cc_name):
    commands = ["cd /home/charles/Project/Blockchain/fabric-samples/test-network && ./network.sh down",
                "cd /home/charles/Project/Blockchain/fabric-samples/test-network && ./network.sh up createChannel",
                # "cd /home/charles/Project/Blockchain/fabric-samples/test-network && source /etc/profile && ./network.sh deployCC -ccn simple -ccp ../../caliper-benchmarks/src/fabric/scenario/simple/go -ccl go",
                "cd /home/charles/Project/Blockchain/fabric-samples/test-network && source /etc/profile && ./network.sh deployCC -ccn "
                + cc_name +
                " -ccp ../../caliper-benchmarks/src/fabric/scenario/"
                + cc_name +
                "/go -ccl go"]
    run_command_and_log(ssh_client, commands, "fabric_log")


def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)
