import argparse

parser = argparse.ArgumentParser()
parser.add_argument("echo", help="fill in this", type=int)
parser.add_argument("--swich_on", help="if swiidfjaifjaois", type=str)
arg = parser.parse_args()
print("sss")
print(arg.echo **2)
print("done")
if arg.swich_on:
    print(arg.swich_on)
