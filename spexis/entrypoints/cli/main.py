# SPDX-License-Identifier: Apache-2.0

# The CLI entrypoint to vLLM.
import signal
import sys

import spexis.entrypoints.cli.benchmark.main
import spexis.entrypoints.cli.openai
import spexis.entrypoints.cli.serve
import spexis.version
from spexis.entrypoints.utils import cli_env_setup
from spexis.utils import FlexibleArgumentParser

CMD_MODULES = [
    spexis.entrypoints.cli.openai,
    spexis.entrypoints.cli.serve,
    spexis.entrypoints.cli.benchmark.main,
]


def register_signal_handlers():

    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)


def main():
    cli_env_setup()

    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=spexis.version.__version__)
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
