# SPDX-License-Identifier: Apache-2.0

from spexis.entrypoints.cli.main import main as vllm_main
from spexis.logger import init_logger

logger = init_logger(__name__)


# Backwards compatibility for the move from vllm.scripts to
# vllm.entrypoints.cli.main
def main():
    logger.warning("spexis.scripts.main() is deprecated. Please "
                   "use spexis.entrypoints.cli.main.main() instead.")
    vllm_main()
