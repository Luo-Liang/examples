import argparse
import paramiko

from subprocess import call

parser = argparse.ArgumentParser(description='PyTorch ImageNet Distributed Launcher')
parser.add_argument('--host', type=str, default='hosts')
parser.add_argument('--port', type=int, default=12345)
parser.add_argument('--network', type=str, default='resnet50')
parser.add_argument('--dir', type=str, default='/home/ubuntu/pytorch-data/tiny-imagenet-200')

args = parser.parse_args()
with open(args.host) as f:
    contents = [x.strip() for x in f.readlines()]
    pass

parameters = ["-a %s --dist-url 'tcp://%s:%s' --dist-backend 'nccl' --multiprocessing-distributed --world-size %s --rank %s" % (args.network, contents[x], args.port, len(contents), x) for x in range(len(contents))]

def LaunchSSHClients(ip):

    # print("connecting to %s" % ip)
    # print("using key file %s" % args.keyFile)
    k = paramiko.RSAKey.from_private_key_file("../large-mls.pem")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #client.set_combine_stderr(True)
    # client.load_system_host_keys("/home/ubuntu/.ssh/large-mls.pem")
    # client.load_host_keys("/home/ubuntu/.ssh/large-mls.pem")

    # client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect(hostname=ip,
                   port=22,
                   username="ubuntu",
                   pkey=k,
                   timeout=1,
                   auth_timeout=1)
    return client

channels = []
for x in range(len(contents)):
    client = LaunchSSHClients(contents[x])
    transport = client.get_transport()
    channel = transport.open_session()
    channel.set_combine_stderr(True)
    channel.exec_command(parameters[x])
    channels.append(channel)
    pass

while all(x.closed for x in channels) == False:
    signal = channels[0].recv(8192)
    print(signal)
    pass




